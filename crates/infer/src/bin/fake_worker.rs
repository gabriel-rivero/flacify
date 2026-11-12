//! A stand-in for `tools/worker.py`, so the client's tests need no Python, no torch and
//! no weights. It speaks the same binary protocol and echoes each window back.
//!
//! It is declared as a `[[bin]]` in this (unpublished) crate because a test binary must
//! exist at build time: `tests/python_worker.rs` spawns it through
//! `CARGO_BIN_EXE_fake-flashsr-worker`. Nothing else ships it.
//!
//! Flags, anywhere on the command line (everything `PythonWorker` passes for the real
//! worker is ignored):
//!
//! * `--fail-after <n>`: exit with status 7 once `n` windows have been served.
//! * `--garbage`: answer every request with the wrong reply magic.
//! * `--corrupt <field>`: mangle one reply field (`status`, `shape`, `length`, `id`),
//!   so the client's protocol checks can be tested without a hostile worker.

use std::io::{self, Read, Write};
use std::process::ExitCode;

const REQUEST_MAGIC: [u8; 4] = *b"FLW1";
const REPLY_MAGIC: [u8; 4] = *b"FLR1";
const HEADER_LEN: usize = 28;

const CMD_HELLO: u8 = 1;
const CMD_WINDOW: u8 = 2;
const CMD_SHUTDOWN: u8 = 3;

const STATUS_OK: u8 = 0;
const STATUS_ERROR: u8 = 1;

/// The window the handshake advertises; the tests mirror these four values.
const WINDOW: u32 = 1024;
const SAMPLE_RATE: u32 = 48_000;
const NOISE_LEN_STEREO: u32 = 8;
const LATENT_SHAPE_STEREO: [u32; 3] = [2, 2, 2];

/// The status the fake exits with when `--fail-after` has been reached.
const CRASH_CODE: u8 = 7;

/// A window body larger than this is a corrupt header, not a request.
const MAX_BODY: usize = 64 * 1024 * 1024;

fn main() -> ExitCode {
    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(reason) => {
            eprintln!("fake worker: {reason}");
            return ExitCode::from(2);
        }
    };
    eprintln!("fake worker: ready in {options:?}");
    match serve(&options) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("fake worker: {err}");
            ExitCode::from(2)
        }
    }
}

#[derive(Debug)]
struct Options {
    fail_after: Option<u32>,
    is_garbage: bool,
    corrupt: Option<Corrupt>,
}

/// One field of a window reply that the fake can get wrong on purpose.
#[derive(Debug, Clone, Copy)]
enum Corrupt {
    Status,
    Shape,
    Length,
    Id,
}

impl Corrupt {
    fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "status" => Ok(Self::Status),
            "shape" => Ok(Self::Shape),
            "length" => Ok(Self::Length),
            "id" => Ok(Self::Id),
            other => Err(format!(
                "--corrupt {other} is not one of status, shape, length, id"
            )),
        }
    }
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Options, String> {
    let mut args = args;
    let mut options = Options {
        fail_after: None,
        is_garbage: false,
        corrupt: None,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--garbage" => options.is_garbage = true,
            "--fail-after" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--fail-after needs a window count".to_string())?;
                let limit = raw
                    .parse::<u32>()
                    .map_err(|_| format!("--fail-after {raw} is not a window count"))?;
                options.fail_after = Some(limit);
            }
            "--corrupt" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--corrupt needs a field name".to_string())?;
                options.corrupt = Some(Corrupt::from_name(&raw)?);
            }
            // The real worker's arguments (`--model-dir`, `--device`, `--flashsr-dir`)
            // come through here; a fake model has no use for them.
            _ => {}
        }
    }
    Ok(options)
}

fn serve(options: &Options) -> io::Result<u8> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let magic: &[u8; 4] = if options.is_garbage {
        b"JUNK"
    } else {
        &REPLY_MAGIC
    };

    let mut windows = 0u32;
    loop {
        let Some(header) = read_header(&mut reader)? else {
            return Ok(0);
        };
        let command = header[4];
        let id = read_u64(&header, 8);
        let channels = read_u32(&header, 16);
        let frames = read_u32(&header, 20);
        let noise_len = read_u32(&header, 24);

        match command {
            CMD_HELLO => {
                Reply {
                    magic,
                    command,
                    status: STATUS_OK,
                    id,
                    channels: 0,
                    frames: 0,
                    payload: hello_json().as_bytes(),
                }
                .write(&mut writer)?;
            }
            CMD_WINDOW => {
                if options.fail_after.is_some_and(|limit| windows >= limit) {
                    eprintln!(
                        "fake worker: exiting after {windows} window(s) with status {CRASH_CODE}"
                    );
                    return Ok(CRASH_CODE);
                }
                let mut input = vec![0u8; body_len(channels, frames)?];
                // The noise is one sample per `noise_len`, without a frame axis.
                let mut noise = vec![0u8; body_len(noise_len, 1)?];
                reader.read_exact(&mut input)?;
                reader.read_exact(&mut noise)?;
                // A copy, i.e. scale by 1.0, so a test can assert exact fidelity.
                let reply = match options.corrupt {
                    None => Reply {
                        magic,
                        command,
                        status: STATUS_OK,
                        id,
                        channels,
                        frames,
                        payload: &input,
                    },
                    Some(Corrupt::Status) => Reply {
                        magic,
                        command,
                        status: STATUS_ERROR,
                        id,
                        channels: 0,
                        frames: 0,
                        payload: b"window refused by the fake",
                    },
                    Some(Corrupt::Shape) => Reply {
                        magic,
                        command,
                        status: STATUS_OK,
                        id,
                        channels: channels + 1,
                        frames,
                        payload: &input,
                    },
                    // Four bytes short: the declared length matches what is sent, so the
                    // client reads a whole payload and still has to reject the length.
                    Some(Corrupt::Length) => Reply {
                        magic,
                        command,
                        status: STATUS_OK,
                        id,
                        channels,
                        frames,
                        payload: &input[..input.len().saturating_sub(4)],
                    },
                    Some(Corrupt::Id) => Reply {
                        magic,
                        command,
                        status: STATUS_OK,
                        id: id.wrapping_add(1),
                        channels,
                        frames,
                        payload: &input,
                    },
                };
                reply.write(&mut writer)?;
                windows += 1;
            }
            CMD_SHUTDOWN => {
                eprintln!("fake worker: shutdown after {windows} window(s)");
                return Ok(0);
            }
            other => {
                let body = format!("unknown command {other}");
                Reply {
                    magic,
                    command: other,
                    status: STATUS_ERROR,
                    id,
                    channels: 0,
                    frames: 0,
                    payload: body.as_bytes(),
                }
                .write(&mut writer)?;
            }
        }
    }
}

/// The handshake, as `tools/worker.py` would write it.
fn hello_json() -> String {
    let shape: Vec<String> = LATENT_SHAPE_STEREO.iter().map(u32::to_string).collect();
    format!(
        "{{\"protocol\": 1, \"device\": \"fake\", \"torch\": \"0\", \"window\": {WINDOW}, \
         \"sample_rate\": {SAMPLE_RATE}, \"noise_len_stereo\": {NOISE_LEN_STEREO}, \
         \"latent_shape_stereo\": [{}], \"load_secs\": 0.0, \"warmup_secs\": 0.0}}",
        shape.join(", ")
    )
}

fn read_header(reader: &mut impl Read) -> io::Result<Option<[u8; HEADER_LEN]>> {
    let mut header = [0u8; HEADER_LEN];
    match reader.read_exact(&mut header) {
        Ok(()) if header[0..4] == REQUEST_MAGIC => Ok(Some(header)),
        Ok(()) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad request magic {:02x?}", &header[0..4]),
        )),
        // A closed pipe and a truncated header both end the session.
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(err),
    }
}

/// One reply header, before it becomes 28 bytes.
struct Reply<'a> {
    magic: &'a [u8; 4],
    command: u8,
    status: u8,
    id: u64,
    channels: u32,
    frames: u32,
    payload: &'a [u8],
}

impl Reply<'_> {
    fn write(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut header = [0u8; HEADER_LEN];
        header[0..4].copy_from_slice(self.magic);
        header[4] = self.command;
        header[5] = self.status;
        header[8..16].copy_from_slice(&self.id.to_le_bytes());
        header[16..20].copy_from_slice(&self.channels.to_le_bytes());
        header[20..24].copy_from_slice(&self.frames.to_le_bytes());
        header[24..28].copy_from_slice(&(self.payload.len() as u32).to_le_bytes());
        writer.write_all(&header)?;
        writer.write_all(self.payload)?;
        writer.flush()
    }
}

/// Bytes for a `channels`-by-`frames` window of `f32`, refusing a size that would
/// allocate the machine away.
fn body_len(channels: u32, frames: u32) -> io::Result<usize> {
    let samples = (channels as usize)
        .checked_mul(frames as usize)
        .ok_or_else(|| invalid_data("window shape overflows"))?;
    let bytes = samples
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid_data("window shape overflows"))?;
    if bytes > MAX_BODY {
        return Err(invalid_data(format!(
            "window body of {bytes} bytes is over the {MAX_BODY} limit"
        )));
    }
    Ok(bytes)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_args(args.iter().map(|arg| (*arg).to_string()))
    }

    #[test]
    fn given_the_arguments_the_client_sends_when_parsed_then_the_fake_takes_no_option() {
        let options = parse(&[
            "tools/worker.py",
            "--model-dir",
            "weights",
            "--device",
            "mps",
            "--flashsr-dir",
            "checkout",
        ])
        .unwrap();
        assert!(options.fail_after.is_none());
        assert!(options.corrupt.is_none());
        assert!(!options.is_garbage);
    }

    #[test]
    fn given_a_non_numeric_window_count_when_parsed_then_it_is_rejected() {
        let err = parse(&["--fail-after", "soon"]).unwrap_err();
        assert!(err.contains("soon"), "{err}");
    }

    #[test]
    fn given_an_unknown_corrupt_field_when_parsed_then_it_is_rejected() {
        let err = parse(&["--corrupt", "everything"]).unwrap_err();
        assert!(err.contains("everything"), "{err}");
    }
}
