//! The Python sidecar: spawn `tools/worker.py` and serve one window per request.
//!
//! The child loads FlashSR once and answers binary requests on stdin and stdout
//! (`docs/decisions.md`, ADR-010). Its log streams into a bounded tail on a background
//! thread, so a chatty worker cannot block on a full pipe. A pipe has no read timeout,
//! so a wedged worker blocks its caller, who drops it and spawns another.

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use flacify_core::{Error, Result};

use crate::model::Model;

/// Request magic; replies carry [`REPLY_MAGIC`].
const REQUEST_MAGIC: [u8; 4] = *b"FLW1";
const REPLY_MAGIC: [u8; 4] = *b"FLR1";

/// Both headers are 28 bytes, little-endian, with fixed padding.
const HEADER_LEN: usize = 28;

const CMD_HELLO: u8 = 1;
const CMD_WINDOW: u8 = 2;
const CMD_SHUTDOWN: u8 = 3;

const STATUS_OK: u8 = 0;

/// The version this client speaks; the handshake must report the same.
const PROTOCOL_VERSION: usize = 1;

/// Lines of the child's stderr kept for diagnostics.
const STDERR_LINES: usize = 20;

/// A reply larger than this is corrupt rather than a window, so refuse it before
/// allocating the buffer.
const MAX_PAYLOAD: usize = 64 * 1024 * 1024;

/// How long a failed request waits for the child's exit status.
const REAP_BUDGET: Duration = Duration::from_millis(250);

/// How long `shutdown` waits for a graceful exit before it kills the child.
const STOP_BUDGET: Duration = Duration::from_secs(5);

/// How often the bounded waits poll the child.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Where the interpreter, the worker script and the weights live.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// The Python interpreter that runs `worker_script`.
    pub program: PathBuf,
    /// `tools/worker.py`.
    pub worker_script: PathBuf,
    /// Directory holding the FlashSR weights.
    pub model_dir: PathBuf,
    /// The FlashSR package checkout, when it is not installed in the interpreter.
    pub flashsr_dir: Option<PathBuf>,
    /// `"mps"` or `"cpu"`; the worker falls back to CPU when Metal is unavailable.
    pub device: String,
}

/// What the loaded model reported at startup.
#[derive(Debug, Clone, PartialEq)]
pub struct Hello {
    /// The device the model actually runs on, after the worker's own fallback.
    pub device: String,
    /// The torch version the worker imported.
    pub torch: String,
    /// Frames per window, which is also the shape `noise_len` specialises for.
    pub window: usize,
    /// Sample rate the model was trained at.
    pub sample_rate: u32,
    /// Noise samples the model's latent grid needs for a stereo window.
    pub noise_len_stereo: usize,
    /// The latent grid's shape itself, for diagnostics.
    pub latent_shape_stereo: Vec<usize>,
    /// Seconds the worker spent loading the weights.
    pub load_secs: f64,
    /// Seconds the worker spent compiling the kernels for the first window.
    pub warmup_secs: f64,
}

impl Hello {
    /// Parse the handshake reply. Every field is required, and the protocol must match.
    fn from_json(text: &str) -> std::result::Result<Hello, String> {
        let mut fields = json::parse_object(text)?;
        let protocol = fields.take_index("protocol")?;
        if protocol != PROTOCOL_VERSION {
            return Err(format!(
                "the worker speaks protocol {protocol}; this client speaks {PROTOCOL_VERSION}"
            ));
        }
        let sample_rate = fields.take_index("sample_rate")?;
        Ok(Hello {
            device: fields.take_text("device")?,
            torch: fields.take_text("torch")?,
            window: fields.take_index("window")?,
            sample_rate: u32::try_from(sample_rate)
                .map_err(|_| format!("sample_rate {sample_rate} is out of range"))?,
            noise_len_stereo: fields.take_index("noise_len_stereo")?,
            latent_shape_stereo: fields.take_indices("latent_shape_stereo")?,
            load_secs: fields.take_number("load_secs")?,
            warmup_secs: fields.take_number("warmup_secs")?,
        })
    }
}

/// A resident `tools/worker.py` behind the [`Model`] trait.
pub struct PythonWorker {
    connection: Connection,
    hello: Hello,
}

impl PythonWorker {
    /// Start the worker and complete the handshake.
    ///
    /// This blocks for as long as the child takes to load the weights, which is several
    /// seconds on the real worker; only its HELLO reply ends the wait.
    pub fn spawn(config: &WorkerConfig) -> Result<Self> {
        Self::spawn_with_args(config, &[])
    }

    /// [`spawn`](Self::spawn) with extra command-line arguments for the child.
    ///
    /// The real worker accepts no others; this exists for the fake worker that the tests
    /// drive, and for one-off diagnostics.
    pub fn spawn_with_args(config: &WorkerConfig, extra_args: &[String]) -> Result<Self> {
        let mut connection = Connection::spawn(config, extra_args)?;
        let hello = connection.hello()?;
        Ok(Self { connection, hello })
    }

    /// What the model reported at startup.
    pub fn hello(&self) -> &Hello {
        &self.hello
    }

    /// The last lines the child logged, oldest first.
    pub fn stderr_tail(&self) -> String {
        self.connection.stderr_tail()
    }

    /// Ask the child to exit, then wait for it. A worker that ignores the request is
    /// killed, so this call cannot hang on a wedged model.
    pub fn shutdown(self) {
        self.connection.stop();
    }
}

impl Model for PythonWorker {
    fn run(
        &mut self,
        input: &[f32],
        noise: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let expected_input = channels * frames;
        if input.len() != expected_input {
            return Err(Error::Inference(format!(
                "input has {} samples but a {channels}x{frames} window needs {expected_input}",
                input.len()
            )));
        }
        let expected_noise = self.noise_len(channels, frames);
        if noise.len() != expected_noise {
            return Err(Error::Inference(format!(
                "noise has {} samples but a {channels}x{frames} window needs {expected_noise}",
                noise.len()
            )));
        }

        let request = Request {
            command: CMD_WINDOW,
            id: self.connection.take_id(),
            channels: as_u32(channels, "channels")?,
            frames: as_u32(frames, "frames")?,
            noise_len: as_u32(expected_noise, "noise length")?,
        };
        let mut body = Vec::with_capacity((input.len() + noise.len()) * size_of::<f32>());
        encode_f32(input, &mut body);
        encode_f32(noise, &mut body);

        let (reply, payload) = self.connection.exchange(&request, &body)?;
        if reply.status != STATUS_OK {
            return Err(Error::Inference(format!(
                "the worker rejected window {}: {}",
                request.id,
                payload_message(&payload)
            )));
        }
        check_window_reply(&reply, &request)?;
        Ok(decode_f32(&payload))
    }

    /// The model's latent grid for its trained stereo window, and one sample per audio
    /// sample otherwise. The grid is a property of the model, which is why the count
    /// comes from the handshake and not from a constant here.
    fn noise_len(&self, channels: usize, frames: usize) -> usize {
        if channels == 2 && frames == self.hello.window {
            self.hello.noise_len_stereo
        } else {
            channels * frames
        }
    }

    /// Measured on the published checkpoint with a band-limited burst at two positions
    /// inside one window: the peak lands 12 samples late both times. The energy centroid
    /// says 24, which is the expected disagreement for an asymmetric response kernel; the
    /// peak is the moment the transient actually arrives, so that is the value used.
    ///
    /// This is a 0.25 ms offset, well under anything audible in a switch, and it is not
    /// what made the low-band splice fail. Re-measure if the weights move.
    fn latency_samples(&self) -> usize {
        12
    }

    fn name(&self) -> &str {
        "flashsr-sidecar"
    }
}

/// One request header before it becomes 28 bytes.
struct Request {
    command: u8,
    id: u64,
    channels: u32,
    frames: u32,
    noise_len: u32,
}

impl Request {
    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut header = [0u8; HEADER_LEN];
        header[0..4].copy_from_slice(&REQUEST_MAGIC);
        header[4] = self.command;
        header[8..16].copy_from_slice(&self.id.to_le_bytes());
        header[16..20].copy_from_slice(&self.channels.to_le_bytes());
        header[20..24].copy_from_slice(&self.frames.to_le_bytes());
        header[24..28].copy_from_slice(&self.noise_len.to_le_bytes());
        header
    }
}

/// One reply header, as the child sent it.
struct Reply {
    command: u8,
    status: u8,
    id: u64,
    channels: u32,
    frames: u32,
    payload_len: u32,
}

impl Reply {
    fn decode(header: &[u8]) -> std::result::Result<Reply, String> {
        if header.len() != HEADER_LEN {
            return Err(format!(
                "a reply header is {HEADER_LEN} bytes, got {}",
                header.len()
            ));
        }
        let magic = &header[0..4];
        if magic != REPLY_MAGIC {
            return Err(format!("bad reply magic {magic:02x?}"));
        }
        Ok(Reply {
            command: header[4],
            status: header[5],
            id: read_u64(header, 8),
            channels: read_u32(header, 16),
            frames: read_u32(header, 20),
            payload_len: read_u32(header, 24),
        })
    }
}

/// The pipes and the child behind them, without any model semantics.
struct Connection {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: ChildStdout,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    stderr_thread: Option<JoinHandle<()>>,
    next_id: u64,
    /// Set once the child has been reaped, so `Drop` does not reap it again.
    has_exited: bool,
}

impl Connection {
    fn spawn(config: &WorkerConfig, extra_args: &[String]) -> Result<Self> {
        let mut command = Command::new(&config.program);
        command
            .arg(&config.worker_script)
            .arg("--model-dir")
            .arg(&config.model_dir)
            .arg("--device")
            .arg(&config.device)
            .args(extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = &config.flashsr_dir {
            command.arg("--flashsr-dir").arg(dir);
        }

        let mut child = command.spawn().map_err(|err| {
            Error::Inference(format!(
                "could not start {} {}: {err}",
                config.program.display(),
                config.worker_script.display()
            ))
        })?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(Error::Inference(
                "the worker's stdin, stdout and stderr were not all piped".to_string(),
            ));
        };

        let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_LINES)));
        let reader_tail = Arc::clone(&stderr_tail);
        let stderr_thread = match thread::Builder::new()
            .name("flashsr-stderr".to_string())
            .spawn(move || drain_stderr(stderr, &reader_tail))
        {
            Ok(handle) => handle,
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Inference(format!(
                    "could not start the stderr reader for the worker: {err}"
                )));
            }
        };

        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr_tail,
            stderr_thread: Some(stderr_thread),
            next_id: 0,
            has_exited: false,
        })
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }

    fn hello(&mut self) -> Result<Hello> {
        let request = Request {
            command: CMD_HELLO,
            id: self.take_id(),
            channels: 0,
            frames: 0,
            noise_len: 0,
        };
        let (reply, payload) = self.exchange(&request, &[])?;
        if reply.status != STATUS_OK {
            return Err(Error::Inference(format!(
                "the worker rejected the handshake: {}",
                payload_message(&payload)
            )));
        }
        Hello::from_json(&String::from_utf8_lossy(&payload))
            .map_err(|reason| Error::Inference(format!("bad handshake reply: {reason}")))
    }

    /// Send one request and read its reply, checking that the reply belongs to it.
    fn exchange(&mut self, request: &Request, body: &[u8]) -> Result<(Reply, Vec<u8>)> {
        self.check_alive()?;
        self.send(request, body)?;

        let header = self.read_bytes(HEADER_LEN, "reading a reply header")?;
        let reply = match Reply::decode(&header) {
            Ok(reply) => reply,
            Err(reason) => {
                return Err(Error::Inference(format!("{reason}{}", self.stderr_note())));
            }
        };
        if reply.command != request.command {
            return Err(Error::Inference(format!(
                "the worker answered command {} for command {}",
                reply.command, request.command
            )));
        }
        if reply.id != request.id {
            return Err(Error::Inference(format!(
                "the worker answered request {} for request {}",
                reply.id, request.id
            )));
        }
        let payload_len = reply.payload_len as usize;
        if payload_len > MAX_PAYLOAD {
            return Err(Error::Inference(format!(
                "the worker announced a {payload_len}-byte reply; the limit is {MAX_PAYLOAD}"
            )));
        }
        let payload = self.read_bytes(payload_len, "reading a reply payload")?;
        Ok((reply, payload))
    }

    fn send(&mut self, request: &Request, body: &[u8]) -> Result<()> {
        let header = request.encode();
        let outcome = match &mut self.stdin {
            Some(stdin) => stdin
                .write_all(&header)
                .and_then(|()| stdin.write_all(body))
                .and_then(|()| stdin.flush()),
            None => {
                return Err(Error::Inference(format!(
                    "the worker's stdin is already closed{}",
                    self.stderr_note()
                )))
            }
        };
        match outcome {
            Ok(()) => Ok(()),
            Err(err) => Err(self.pipe_failure("writing a request", err)),
        }
    }

    fn read_bytes(&mut self, len: usize, what: &str) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        let outcome = self.stdout.read_exact(&mut bytes);
        match outcome {
            Ok(()) => Ok(bytes),
            Err(err) => Err(self.pipe_failure(what, err)),
        }
    }

    /// Fail before writing when the child is already gone; the caller restarts the worker.
    fn check_alive(&mut self) -> Result<()> {
        let outcome = self.child.try_wait();
        match outcome {
            Ok(None) => Ok(()),
            Ok(Some(status)) => {
                self.wrap_up();
                Err(Error::Inference(format!(
                    "the worker exited before the request: {}{}",
                    exit_label(status),
                    self.stderr_note()
                )))
            }
            Err(err) => Err(Error::Inference(format!(
                "could not check the worker process: {err}{}",
                self.stderr_note()
            ))),
        }
    }

    /// A pipe failed: the reply will never arrive, so the caller starts a new worker.
    fn pipe_failure(&mut self, what: &str, err: io::Error) -> Error {
        let status = self.wait_for_exit(REAP_BUDGET);
        let fate = match status {
            Some(status) => {
                self.wrap_up();
                format!("the worker exited: {}", exit_label(status))
            }
            None => "the worker is still running".to_string(),
        };
        Error::Inference(format!(
            "{what} failed: {err}; {fate}{}",
            self.stderr_note()
        ))
    }

    /// Poll the child for its exit status, for at most `budget`.
    fn wait_for_exit(&mut self, budget: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + budget;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return Some(status),
                Ok(None) if Instant::now() < deadline => thread::sleep(POLL_INTERVAL),
                _ => return None,
            }
        }
    }

    /// Join the stderr reader once the child is gone, so the tail covers its last words.
    fn wrap_up(&mut self) {
        self.has_exited = true;
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }

    fn stderr_tail(&self) -> String {
        match self.stderr_tail.lock() {
            Ok(tail) => tail.iter().cloned().collect::<Vec<_>>().join("\n"),
            Err(_) => "the stderr reader panicked; no log available".to_string(),
        }
    }

    fn stderr_note(&self) -> String {
        let tail = self.stderr_tail();
        if tail.is_empty() {
            String::new()
        } else {
            format!("\nworker stderr (last lines):\n{tail}")
        }
    }

    fn stop(mut self) {
        let request = Request {
            command: CMD_SHUTDOWN,
            id: 0,
            channels: 0,
            frames: 0,
            noise_len: 0,
        };
        if let Some(mut stdin) = self.stdin.take() {
            let _ = stdin.write_all(&request.encode());
            let _ = stdin.flush();
        }
        if self.wait_for_exit(STOP_BUDGET).is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        self.wrap_up();
    }
}

impl Drop for Connection {
    /// A dropped worker must not keep the pipes, or 3 GB of weights on Metal, alive.
    fn drop(&mut self) {
        if !self.has_exited {
            self.stdin = None;
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

/// Keep the last [`STDERR_LINES`] lines, and never stop reading: the child blocks on a
/// full pipe otherwise.
fn drain_stderr(stderr: std::process::ChildStderr, tail: &Mutex<VecDeque<String>>) {
    let mut reader = BufReader::new(stderr);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => return,
            Ok(_) => {
                let text = String::from_utf8_lossy(&line).trim_end().to_string();
                if let Ok(mut tail) = tail.lock() {
                    if tail.len() >= STDERR_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(text);
                }
            }
            Err(_) => return,
        }
    }
}

/// Check that a window reply has the shape of the window that was asked for.
/// `exit code 7` on every platform. `ExitStatus` renders as "exit status: 7" on unix
/// and "exit code: 7" on Windows, and this string ends up in a user-facing error.
fn exit_label(status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exit code {code}"),
        None => "killed by a signal".to_string(),
    }
}

fn check_window_reply(reply: &Reply, request: &Request) -> Result<()> {
    if reply.channels != request.channels || reply.frames != request.frames {
        return Err(Error::Inference(format!(
            "the worker answered {}x{} for a {}x{} window",
            reply.channels, reply.frames, request.channels, request.frames
        )));
    }
    let expected = request.channels as usize * request.frames as usize * size_of::<f32>();
    if reply.payload_len as usize != expected {
        return Err(Error::Inference(format!(
            "the worker sent {} bytes for a {}x{} window; {expected} expected",
            reply.payload_len, request.channels, request.frames
        )));
    }
    Ok(())
}

fn as_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::Inference(format!(
            "{what} of {value} does not fit the protocol's 32-bit field"
        ))
    })
}

fn encode_f32(samples: &[f32], out: &mut Vec<u8>) {
    out.reserve(std::mem::size_of_val(samples));
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(word)
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(word)
}

/// The worker's error message, or a placeholder when it sent none.
fn payload_message(payload: &[u8]) -> String {
    if payload.is_empty() {
        return "(no message)".to_string();
    }
    String::from_utf8_lossy(payload).into_owned()
}

/// A reader for the handshake's flat JSON object.
///
/// `serde_json` is only a dev-dependency of this crate, and the handshake is eight
/// scalars written by our own worker, so the client parses what it needs itself.
mod json {
    /// One value from the handshake object.
    pub(super) enum Value {
        Text(String),
        Number(f64),
        List(Vec<f64>),
    }

    /// The object's key/value pairs, with typed accessors that consume their key.
    pub(super) struct Fields {
        pairs: Vec<(String, Value)>,
    }

    pub(super) fn parse_object(text: &str) -> Result<Fields, String> {
        let mut parser = Parser {
            chars: text.chars().peekable(),
        };
        let pairs = parser.object()?;
        parser.skip_space();
        if parser.chars.next().is_some() {
            return Err("trailing text after the JSON object".to_string());
        }
        Ok(Fields { pairs })
    }

    impl Fields {
        fn take(&mut self, key: &str) -> Option<Value> {
            let index = self.pairs.iter().position(|(name, _)| name == key)?;
            Some(self.pairs.remove(index).1)
        }

        pub(super) fn take_text(&mut self, key: &str) -> Result<String, String> {
            match self.take(key) {
                Some(Value::Text(text)) => Ok(text),
                Some(_) => Err(format!("{key:?} is not a string")),
                None => Err(format!("the handshake reply has no {key:?} field")),
            }
        }

        pub(super) fn take_number(&mut self, key: &str) -> Result<f64, String> {
            match self.take(key) {
                Some(Value::Number(number)) => Ok(number),
                Some(_) => Err(format!("{key:?} is not a number")),
                None => Err(format!("the handshake reply has no {key:?} field")),
            }
        }

        pub(super) fn take_index(&mut self, key: &str) -> Result<usize, String> {
            let number = self.take_number(key)?;
            whole_number(key, number)
        }

        pub(super) fn take_indices(&mut self, key: &str) -> Result<Vec<usize>, String> {
            match self.take(key) {
                Some(Value::List(items)) => items
                    .into_iter()
                    .map(|number| whole_number(key, number))
                    .collect(),
                Some(_) => Err(format!("{key:?} is not a list")),
                None => Err(format!("the handshake reply has no {key:?} field")),
            }
        }
    }

    fn whole_number(key: &str, number: f64) -> Result<usize, String> {
        if number < 0.0 || number > usize::MAX as f64 || number.fract() != 0.0 {
            return Err(format!("{key:?} is not a whole number: {number}"));
        }
        Ok(number as usize)
    }

    struct Parser<'a> {
        chars: std::iter::Peekable<std::str::Chars<'a>>,
    }

    impl Parser<'_> {
        fn skip_space(&mut self) {
            while matches!(self.chars.peek(), Some(c) if c.is_ascii_whitespace()) {
                self.chars.next();
            }
        }

        fn object(&mut self) -> Result<Vec<(String, Value)>, String> {
            self.skip_space();
            if self.chars.next() != Some('{') {
                return Err("the handshake reply is not a JSON object".to_string());
            }
            let mut pairs = Vec::new();
            self.skip_space();
            if self.chars.peek() == Some(&'}') {
                self.chars.next();
                return Ok(pairs);
            }
            loop {
                self.skip_space();
                let key = self.string()?;
                self.skip_space();
                if self.chars.next() != Some(':') {
                    return Err(format!("expected ':' after the key {key:?}"));
                }
                self.skip_space();
                let value = self.value()?;
                pairs.push((key, value));
                self.skip_space();
                match self.chars.next() {
                    Some(',') => continue,
                    Some('}') => return Ok(pairs),
                    _ => return Err("expected ',' or '}' in the handshake reply".to_string()),
                }
            }
        }

        fn value(&mut self) -> Result<Value, String> {
            match self.chars.peek() {
                Some('"') => Ok(Value::Text(self.string()?)),
                Some('[') => Ok(Value::List(self.list()?)),
                Some(c) if *c == '-' || c.is_ascii_digit() => Ok(Value::Number(self.number()?)),
                Some(other) => Err(format!("unsupported JSON value starting with {other:?}")),
                None => Err("the handshake reply ends inside a value".to_string()),
            }
        }

        fn list(&mut self) -> Result<Vec<f64>, String> {
            self.chars.next();
            let mut items = Vec::new();
            self.skip_space();
            if self.chars.peek() == Some(&']') {
                self.chars.next();
                return Ok(items);
            }
            loop {
                self.skip_space();
                items.push(self.number()?);
                self.skip_space();
                match self.chars.next() {
                    Some(',') => continue,
                    Some(']') => return Ok(items),
                    _ => return Err("expected ',' or ']' in a JSON list".to_string()),
                }
            }
        }

        fn number(&mut self) -> Result<f64, String> {
            let mut text = String::new();
            while let Some(c) = self.chars.peek() {
                if !c.is_ascii_digit() && !matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                    break;
                }
                text.push(*c);
                self.chars.next();
            }
            text.parse::<f64>()
                .map_err(|_| format!("{text:?} is not a number"))
        }

        fn string(&mut self) -> Result<String, String> {
            if self.chars.next() != Some('"') {
                return Err("expected a JSON string".to_string());
            }
            let mut text = String::new();
            loop {
                match self.chars.next() {
                    None => return Err("the handshake reply ends inside a string".to_string()),
                    Some('"') => return Ok(text),
                    Some('\\') => text.push(self.escape()?),
                    Some(c) => text.push(c),
                }
            }
        }

        fn escape(&mut self) -> Result<char, String> {
            match self.chars.next() {
                Some('"') => Ok('"'),
                Some('\\') => Ok('\\'),
                Some('/') => Ok('/'),
                Some('b') => Ok('\u{8}'),
                Some('f') => Ok('\u{c}'),
                Some('n') => Ok('\n'),
                Some('r') => Ok('\r'),
                Some('t') => Ok('\t'),
                Some('u') => self.unicode_escape(),
                Some(other) => Err(format!("unsupported JSON escape \\{other}")),
                None => Err("the handshake reply ends inside an escape".to_string()),
            }
        }

        fn unicode_escape(&mut self) -> Result<char, String> {
            let high = self.hex4()?;
            let code = if (0xD800..0xDC00).contains(&high) {
                if self.chars.next() != Some('\\') || self.chars.next() != Some('u') {
                    return Err("a JSON surrogate half stands alone".to_string());
                }
                let low = self.hex4()?;
                0x10000 + ((high - 0xD800) << 10) + low.wrapping_sub(0xDC00)
            } else {
                high
            };
            char::from_u32(code).ok_or_else(|| format!("\\u{code:04x} is not a character"))
        }

        fn hex4(&mut self) -> Result<u32, String> {
            let mut value = 0u32;
            for _ in 0..4 {
                let digit = self
                    .chars
                    .next()
                    .ok_or_else(|| "a \\u escape is truncated".to_string())?;
                let nibble = digit
                    .to_digit(16)
                    .ok_or_else(|| format!("{digit:?} is not a hex digit"))?;
                value = value * 16 + nibble;
            }
            Ok(value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HANDSHAKE: &str = concat!(
        r#"{"protocol": 1, "device": "mps", "torch": "2.4.0", "window": 245760, "#,
        r#""sample_rate": 48000, "noise_len_stereo": 1920, "#,
        r#""latent_shape_stereo": [4, 8, 60], "load_secs": 8.2, "warmup_secs": 1.5}"#
    );

    #[test]
    fn given_the_worker_handshake_when_parsed_then_every_field_is_present() {
        let hello = Hello::from_json(HANDSHAKE).unwrap();
        assert_eq!(hello.device, "mps");
        assert_eq!(hello.torch, "2.4.0");
        assert_eq!(hello.window, 245_760);
        assert_eq!(hello.sample_rate, 48_000);
        assert_eq!(hello.noise_len_stereo, 1920);
        assert_eq!(hello.latent_shape_stereo, vec![4, 8, 60]);
        assert_eq!(hello.load_secs, 8.2);
        assert_eq!(hello.warmup_secs, 1.5);
    }

    #[test]
    fn given_another_protocol_when_the_handshake_is_parsed_then_it_is_rejected() {
        let text = HANDSHAKE.replace("\"protocol\": 1", "\"protocol\": 2");
        let err = Hello::from_json(&text).unwrap_err();
        assert!(err.contains("protocol 2"), "{err}");
    }

    #[test]
    fn given_a_missing_field_when_the_handshake_is_parsed_then_the_error_names_it() {
        let text = HANDSHAKE.replace("\"device\": \"mps\",", "");
        let err = Hello::from_json(&text).unwrap_err();
        assert!(err.contains("\"device\""), "{err}");
    }

    #[test]
    fn given_an_escaped_unicode_name_when_the_handshake_is_parsed_then_the_character_survives() {
        let text = HANDSHAKE.replace("\"mps\"", r#""mps \u00e9 \ud83c\udfb5""#);
        let hello = Hello::from_json(&text).unwrap();
        assert_eq!(hello.device, "mps \u{e9} \u{1f3b5}");
    }

    #[test]
    fn given_trailing_text_after_the_object_when_it_is_parsed_then_it_is_rejected() {
        let text = format!("{HANDSHAKE} garbage");
        let err = Hello::from_json(&text).unwrap_err();
        assert!(err.contains("trailing text"), "{err}");
    }

    #[test]
    fn given_a_fractional_index_when_it_is_parsed_then_it_is_rejected() {
        let text = HANDSHAKE.replace("\"window\": 245760", "\"window\": 245760.5");
        let err = Hello::from_json(&text).unwrap_err();
        assert!(err.contains("whole number"), "{err}");
    }
}
