use std::env;
use std::io::{self, Read};
use std::process::{self, Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(200);

enum ParentEvent {
    Disconnected,
    ReadError(io::Error),
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
            exit_with_status(status);
        }

        match parent_rx.recv_timeout(CHILD_POLL_INTERVAL) {
            Ok(ParentEvent::Disconnected) => {
                eprintln!(
                    "[wrapper] parent pipe closed; killing child pid {}",
                    child.id()
                );
                terminate_child(&mut child)?;
                return Ok(());
            }
            Ok(ParentEvent::ReadError(err)) => {
                eprintln!(
                    "[wrapper] parent pipe monitor failed: {err}; killing child pid {}",
                    child.id()
                );
                terminate_child(&mut child)?;
                return Ok(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!(
                    "[wrapper] parent monitor thread exited unexpectedly; killing child pid {}",
                    child.id()
                );
                terminate_child(&mut child)?;
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
