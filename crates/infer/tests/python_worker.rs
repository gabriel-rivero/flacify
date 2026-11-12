//! Drive `PythonWorker` against the fake sidecar, so these tests need no Python, no
//! torch and no weights.
//!
//! The fake worker is `crates/infer/src/bin/fake_worker.rs`; its handshake advertises
//! the window and noise lengths mirrored below.

use std::path::PathBuf;

use flacify_core::Error;
use flacify_infer::model::Model;
use flacify_infer::python::{PythonWorker, WorkerConfig};

const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_fake-flashsr-worker");
const WINDOW: usize = 1024;
const NOISE_LEN_STEREO: usize = 8;

fn config() -> WorkerConfig {
    WorkerConfig {
        program: PathBuf::from(FAKE_WORKER),
        // The fake worker ignores the script and the model paths `spawn` passes.
        worker_script: PathBuf::from("tools/worker.py"),
        model_dir: PathBuf::from("unused"),
        flashsr_dir: Some(PathBuf::from("unused/flashsr")),
        device: "fake".to_string(),
    }
}

fn spawn_with(extra_args: &[&str]) -> PythonWorker {
    let extra: Vec<String> = extra_args.iter().map(|arg| (*arg).to_string()).collect();
    PythonWorker::spawn_with_args(&config(), &extra).expect("the fake worker should start")
}

/// A signal whose samples are all distinct, so a reordered window cannot round trip.
fn signal(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index as f32) * 0.001 - 0.5).collect()
}

fn noise(len: usize) -> Vec<f32> {
    vec![0.25f32; len]
}

#[test]
fn given_the_fake_worker_when_spawned_then_the_handshake_reports_the_protocol_fields() {
    let worker = PythonWorker::spawn(&config()).expect("the fake worker should start");
    let hello = worker.hello();
    assert_eq!(hello.device, "fake");
    assert_eq!(hello.torch, "0");
    assert_eq!(hello.window, WINDOW);
    assert_eq!(hello.sample_rate, 48_000);
    assert_eq!(hello.noise_len_stereo, NOISE_LEN_STEREO);
    assert_eq!(hello.latent_shape_stereo, vec![2, 2, 2]);
    assert_eq!(hello.load_secs, 0.0);
    assert_eq!(hello.warmup_secs, 0.0);
    assert_eq!(worker.name(), "flashsr-sidecar");
    worker.shutdown();
}

#[test]
fn given_a_window_when_run_then_the_samples_round_trip_and_noise_len_follows_the_shape() {
    let mut worker = spawn_with(&[]);

    // The trained stereo window uses the model's latent grid; every other shape falls
    // back to one noise sample per audio sample.
    assert_eq!(worker.noise_len(2, WINDOW), NOISE_LEN_STEREO);
    assert_eq!(worker.noise_len(1, WINDOW), WINDOW);
    assert_eq!(worker.noise_len(2, WINDOW / 2), WINDOW);

    let stereo = signal(2 * WINDOW);
    let out = worker
        .run(&stereo, &noise(NOISE_LEN_STEREO), 2, WINDOW)
        .unwrap();
    assert_eq!(out, stereo);

    let mono = signal(WINDOW / 2);
    let out = worker
        .run(&mono, &noise(WINDOW / 2), 1, WINDOW / 2)
        .unwrap();
    assert_eq!(out, mono);
    worker.shutdown();
}

#[test]
fn given_a_window_of_the_wrong_size_when_run_then_it_is_rejected_before_the_worker_sees_it() {
    let mut worker = spawn_with(&[]);
    let input = signal(2 * WINDOW);
    let noise = noise(NOISE_LEN_STEREO);
    // One sample over, which the fake would answer without complaint if the check
    // were gone: the rejection has to come from the client.
    let mut input_too_long = input.clone();
    input_too_long.push(0.0);
    let mut noise_too_long = noise.clone();
    noise_too_long.push(0.0);

    let err = worker.run(&input_too_long, &noise, 2, WINDOW).unwrap_err();
    assert!(err.to_string().contains("input has 2049 samples"), "{err}");
    let err = worker.run(&input, &noise_too_long, 2, WINDOW).unwrap_err();
    assert!(err.to_string().contains("noise has 9 samples"), "{err}");

    // Nothing reached the pipe: the session still answers the next valid window.
    let out = worker.run(&input, &noise, 2, WINDOW).unwrap();
    assert_eq!(out, input);
    worker.shutdown();
}

#[test]
fn given_the_worker_exits_mid_session_when_the_next_window_runs_then_it_is_an_inference_error() {
    let mut worker = spawn_with(&["--fail-after", "1"]);
    let input = signal(2 * WINDOW);
    let noise = noise(NOISE_LEN_STEREO);

    // The first window is served, then the fake worker exits with status 7.
    let out = worker.run(&input, &noise, 2, WINDOW).unwrap();
    assert_eq!(out, input);

    let err = worker.run(&input, &noise, 2, WINDOW).unwrap_err();
    match &err {
        Error::Inference(message) => {
            assert!(message.contains("exit code 7"), "{message}");
            assert!(message.contains("exiting after 1 window"), "{message}");
        }
        other => panic!("expected an inference error, got {other:?}"),
    }
    assert!(
        worker.stderr_tail().contains("exiting after 1 window"),
        "{}",
        worker.stderr_tail()
    );

    // The failure reaped the child, so a further window fails at the liveness check
    // instead of reaching the pipe.
    let err = worker.run(&input, &noise, 2, WINDOW).unwrap_err();
    assert!(
        err.to_string().contains("exited before the request"),
        "{err}"
    );
    assert!(err.to_string().contains("exit code 7"), "{err}");
    worker.shutdown();
}

#[test]
fn given_a_window_the_worker_refuses_when_run_then_its_message_reaches_the_error() {
    let mut worker = spawn_with(&["--corrupt", "status"]);
    let err = worker
        .run(&signal(2 * WINDOW), &noise(NOISE_LEN_STEREO), 2, WINDOW)
        .unwrap_err();
    match &err {
        Error::Inference(message) => {
            assert!(message.contains("rejected window 1"), "{message}");
            assert!(message.contains("window refused by the fake"), "{message}");
        }
        other => panic!("expected an inference error, got {other:?}"),
    }
    worker.shutdown();
}

#[test]
fn given_a_reply_with_the_wrong_shape_when_run_then_it_is_an_inference_error() {
    let mut worker = spawn_with(&["--corrupt", "shape"]);
    let err = worker
        .run(&signal(2 * WINDOW), &noise(NOISE_LEN_STEREO), 2, WINDOW)
        .unwrap_err();
    assert!(
        err.to_string().contains("answered 3x1024 for a 2x1024"),
        "{err}"
    );
    worker.shutdown();
}

#[test]
fn given_a_reply_with_the_wrong_length_when_run_then_it_is_an_inference_error() {
    let mut worker = spawn_with(&["--corrupt", "length"]);
    let err = worker
        .run(&signal(2 * WINDOW), &noise(NOISE_LEN_STEREO), 2, WINDOW)
        .unwrap_err();
    assert!(
        err.to_string().contains("bytes for a 2x1024 window"),
        "{err}"
    );
    worker.shutdown();
}

#[test]
fn given_a_reply_to_another_request_when_run_then_it_is_an_inference_error() {
    let mut worker = spawn_with(&["--corrupt", "id"]);
    let err = worker
        .run(&signal(2 * WINDOW), &noise(NOISE_LEN_STEREO), 2, WINDOW)
        .unwrap_err();
    assert!(err.to_string().contains("answered request 2"), "{err}");
    worker.shutdown();
}

#[test]
fn given_a_reply_with_a_bad_magic_when_spawning_then_the_handshake_fails() {
    let error = match PythonWorker::spawn_with_args(&config(), &["--garbage".to_string()]) {
        Ok(_) => panic!("the handshake should not succeed"),
        Err(error) => error,
    };
    match &error {
        Error::Inference(message) => assert!(message.contains("bad reply magic"), "{message}"),
        other => panic!("expected an inference error, got {other:?}"),
    }
}
