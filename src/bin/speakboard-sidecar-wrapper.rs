use std::env;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::{self, Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde::Deserialize;

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(200);

enum ParentEvent {
    Disconnected,
    ReadError(io::Error),
}

#[derive(Debug, Default, Deserialize)]
struct ConfigFile {
    socket_path: Option<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[wrapper] fatal error: {err}");
        process::exit(1);
    }
}

fn run() -> io::Result<()> {
    let mut args = env::args_os();
    let _self = args.next();

    let Some(program) = args.next() else {
        eprintln!(
            "[wrapper] usage: speakboard-sidecar-wrapper <program> [args...]"
        );
        process::exit(64);
    };

    let child_args: Vec<_> = args.collect();
    let socket_path = socket_path_from_args(&child_args);

    let mut child = Command::new(&program)
        .args(&child_args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    eprintln!("[wrapper] started child pid {}", child.id());

    let (parent_tx, parent_rx) = mpsc::channel();
    thread::spawn(move || monitor_parent_stdin(parent_tx));

    loop {
        if let Some(status) = child.try_wait()? {
            cleanup_socket_file(socket_path.as_ref());
            exit_with_status(status);
        }

        match parent_rx.recv_timeout(CHILD_POLL_INTERVAL) {
            Ok(ParentEvent::Disconnected) => {
                eprintln!(
                    "[wrapper] parent pipe closed; killing child pid {}",
                    child.id()
                );
                terminate_child(&mut child)?;
                cleanup_socket_file(socket_path.as_ref());
                return Ok(());
            }
            Ok(ParentEvent::ReadError(err)) => {
                eprintln!(
                    "[wrapper] parent pipe monitor failed: {err}; killing child pid {}",
                    child.id()
                );
                terminate_child(&mut child)?;
                cleanup_socket_file(socket_path.as_ref());
                return Ok(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!(
                    "[wrapper] parent monitor thread exited unexpectedly; killing child pid {}",
                    child.id()
                );
                terminate_child(&mut child)?;
                cleanup_socket_file(socket_path.as_ref());
                return Ok(());
            }
        }
    }
}

fn monitor_parent_stdin(tx: mpsc::Sender<ParentEvent>) {
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let mut buf = [0_u8; 1024];

    loop {
        match handle.read(&mut buf) {
            Ok(0) => {
                let _ = tx.send(ParentEvent::Disconnected);
                return;
            }
            Ok(_) => {}
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => {
                let _ = tx.send(ParentEvent::ReadError(err));
                return;
            }
        }
    }
}

fn terminate_child(child: &mut Child) -> io::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }

    child.kill()?;
    let _ = child.wait()?;
    Ok(())
}

fn exit_with_status(status: ExitStatus) -> ! {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;

        if let Some(signal) = status.signal() {
            process::exit(128 + signal);
        }
    }

    process::exit(status.code().unwrap_or(1));
}

fn socket_path_from_args(args: &[std::ffi::OsString]) -> Option<PathBuf> {
    let config_path = parse_config_path(args)?;
    let text = std::fs::read_to_string(config_path).ok()?;
    let cfg: ConfigFile = serde_json::from_str(&text).ok()?;
    cfg.socket_path.map(PathBuf::from)
}

fn parse_config_path(args: &[std::ffi::OsString]) -> Option<&std::ffi::OsStr> {
    let pos = args.iter().position(|arg| arg == "--config")?;
    args.get(pos + 1).map(|arg| arg.as_os_str())
}

fn cleanup_socket_file(path: Option<&PathBuf>) {
    let Some(path) = path else { return };
    if let Err(err) = std::fs::remove_file(path)
        && err.kind() != io::ErrorKind::NotFound
    {
        eprintln!(
            "[wrapper] failed to remove socket file {}: {err}",
            path.display()
        );
    }
}
