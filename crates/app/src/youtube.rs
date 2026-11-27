//! A URL as a source: fetch the audio with `yt-dlp`, then hand the pipeline a file.
//!
//! The pipeline only ever sees a path, so one download resolves to one file and nothing
//! downstream knows where it came from. That keeps the cache, the seek, the frontier and
//! the A/B switch exactly as they are for a local file, and it is why the download is a
//! file rather than a pipe: the source tier reads it once and never again.
//!
//! `yt-dlp` is a user-supplied tool on `PATH`, not a dependency of this repo, so it can
//! update independently. When it is absent the error says how to install it and nothing
//! else happens.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::process::{self, Cancel};

/// Overrides where `yt-dlp` is found, the way `FLACIFY_UV` does for `uv`.
const YTDLP_ENV: &str = "FLACIFY_YTDLP";

/// The container we ask for. FLAC is in `symphonia`'s feature list, is lossless so nothing
/// is thrown away before the model sees it, and is written by the `ffmpeg` yt-dlp already
/// needs. The default (`bestaudio`) is WebM/Opus, which this build cannot decode.
const AUDIO_FORMAT: &str = "flac";

/// The containers the pipeline can actually decode, so a download it cannot read is named
/// here rather than failing later inside the decoder.
const DECODABLE: &[&str] = &["flac", "wav", "m4a", "mp3", "aac", "ogg"];

/// The names a fetch printed, kept beside the audio so a replay shows the same title.
const NAMES_FILE: &str = "names.tsv";

/// One URL, resolved enough to fetch: the address to give yt-dlp and the id to key it by.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub url: String,
    pub video_id: String,
}

/// What a fetch produced: the file to play, and the names to show beside it.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub path: PathBuf,
    pub title: Option<String>,
    pub artist: Option<String>,
}

impl Track {
    /// The name the window and the report lines show, falling back to the file name.
    pub fn label(&self) -> String {
        self.title
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| file_name(&self.path))
    }
}

/// The request a command-line argument names, when it is a URL.
///
/// A path is not a URL and returns `None`, which is what keeps every existing `flacify
/// play song.mp3` unchanged. The id is read from the shapes YouTube uses; anything else
/// keeps its URL and is keyed by a hash of it, because yt-dlp accepts more sites than we
/// know how to parse.
pub fn parse(text: &str) -> Option<Request> {
    let url = text.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return None;
    }
    let video_id = video_id_of(url).unwrap_or_else(|| format!("url-{}", short_hash(url)));
    Some(Request {
        url: url.to_string(),
        video_id,
    })
}

/// The id from a watch, `youtu.be` or `shorts` URL, when the URL is one of those shapes.
fn video_id_of(url: &str) -> Option<String> {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let (host, path) = after_scheme.split_once('/').unwrap_or((after_scheme, ""));
    let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
    let id = if host.ends_with("youtu.be") {
        path.split(['?', '/']).next().unwrap_or("")
    } else if host.ends_with("youtube.com") || host.ends_with("youtube-nocookie.com") {
        if let Some(rest) = path.strip_prefix("shorts/") {
            rest.split(['?', '/']).next().unwrap_or("")
        } else if let Some(rest) = path.strip_prefix("embed/") {
            rest.split(['?', '/']).next().unwrap_or("")
        } else {
            query
                .split('&')
                .find_map(|pair| pair.strip_prefix("v="))
                .unwrap_or("")
        }
    } else {
        ""
    };
    sanitised(id)
}

/// An id that is safe to use as a directory name: the alphabet YouTube uses, and nothing
/// that could climb out of the downloads directory.
fn sanitised(id: &str) -> Option<String> {
    let id = id.trim();
    let safe = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    safe.then(|| id.to_string())
}

/// A short, stable key for a URL we cannot parse an id out of. Not a cache key: only a
/// directory name, so a collision costs a re-download rather than the wrong audio.
fn short_hash(text: &str) -> String {
    let mut hash: u64 = blake3::hash(text.as_bytes()).as_bytes()[..8]
        .iter()
        .fold(0u64, |acc, byte| (acc << 8) | u64::from(*byte));
    hash ^= text.len() as u64;
    format!("{hash:016x}")
}

/// The file a command should actually open: the input itself when it is a path, and a
/// downloaded track when it is a URL.
///
/// This is the whole of what `play` and `enhance` need to accept a URL, and it is why the
/// rest of the pipeline never learns where the audio came from. A local path is returned
/// untouched, so nothing about `flacify play song.mp3` changes.
pub fn resolve(input: &Path, downloads: &Path, cancel: Option<&Cancel>) -> Result<Track> {
    let Some(request) = parse(&input.to_string_lossy()) else {
        return Ok(Track {
            path: input.to_path_buf(),
            title: None,
            artist: None,
        });
    };
    let track = fetch(&request, downloads, cancel)?;
    eprintln!("source    {} ({})", track.label(), request.video_id);
    Ok(track)
}

/// Where `yt-dlp` is: the override, then `PATH`. `None` means the error is worth raising,
/// not that a download should be skipped.
pub fn find_ytdlp() -> Option<PathBuf> {
    if let Some(program) = std::env::var_os(YTDLP_ENV) {
        let path = PathBuf::from(program);
        if path.components().count() > 1 {
            return process::is_executable(&path).then_some(path);
        }
        return process::find_on_path(&path.to_string_lossy());
    }
    process::find_on_path("yt-dlp")
}

/// The file a fetch of `request` produced, or the one an earlier fetch already left.
///
/// A file already in `dir` is trusted: it is keyed by the video id, so another video
/// cannot have written it, and `flacify cache --evict` is how it is reclaimed or
/// refreshed. The second play therefore spawns nothing, and shows the same title because
/// the names are written beside the audio rather than asked for again.
pub fn fetch(request: &Request, dir: &Path, cancel: Option<&Cancel>) -> Result<Track> {
    let target = dir.join(&request.video_id);
    if let Some(existing) = already_downloaded(&target) {
        let names = Names::load(&target);
        return Ok(Track {
            path: existing,
            title: names.title,
            artist: names.artist,
        });
    }

    let program = find_ytdlp().context(
        "yt-dlp is not installed, and it is what fetches a URL. Install it with `brew install \
         yt-dlp`, `pipx install yt-dlp` or `pip install yt-dlp`, then try again.",
    )?;
    std::fs::create_dir_all(&target).with_context(|| format!("creating {}", target.display()))?;

    let template = target.join("%(id)s.%(ext)s");
    let fields = "after_move:%(filepath)s\t%(title)s\t%(artist,uploader,channel)s";
    let template = template.to_string_lossy().into_owned();
    let args = [
        "--no-playlist",
        "--no-warnings",
        "--no-continue",
        "--newline",
        "-x",
        "--audio-format",
        AUDIO_FORMAT,
        "--audio-quality",
        "0",
        "-o",
        &template,
        "--print",
        fields,
        &request.url,
    ];
    let finished = process::run_streaming(&program, &args, cancel, &mut |line| {
        // Downloads report as `[download]  12.3% of ...`. Only progress is shown, so a
        // person watching a terminal sees the fetch move without a line per fragment.
        if line.starts_with("[download]") && line.contains('%') {
            eprint!("\r  {line}");
        }
    })
    .with_context(|| format!("fetching {}", request.url))?;
    eprintln!();

    let line = finished
        .stdout
        .iter()
        .rev()
        .find(|line| {
            line.split('\t')
                .next()
                .is_some_and(|path| !path.trim().is_empty())
        })
        .context("yt-dlp printed no file name, so nothing was written")?;
    let mut parts = line.splitn(3, '\t');
    let path = PathBuf::from(parts.next().unwrap_or("").trim());
    let title = clean(parts.next());
    let artist = clean(parts.next());

    check_decodable(&path).with_context(|| {
        // yt-dlp's own warnings explain most of these, and a report needs them together.
        let note = finished.stderr.trim();
        if note.is_empty() {
            format!("fetching {}", request.url)
        } else {
            format!("fetching {}; yt-dlp said: {note}", request.url)
        }
    })?;
    Names {
        title: title.clone(),
        artist: artist.clone(),
    }
    .save(&target);
    Ok(Track {
        path,
        title,
        artist,
    })
}

/// The names a fetch printed, saved beside the audio. Not a cache key: it is only what
/// the window shows, and losing it costs a title line rather than the track.
#[derive(Default)]
struct Names {
    title: Option<String>,
    artist: Option<String>,
}

impl Names {
    fn load(dir: &Path) -> Names {
        let Ok(text) = std::fs::read_to_string(dir.join(NAMES_FILE)) else {
            return Names::default();
        };
        let mut names = Names::default();
        for line in text.lines() {
            if let Some(title) = line.strip_prefix("title\t") {
                names.title = clean(Some(title));
            } else if let Some(artist) = line.strip_prefix("artist\t") {
                names.artist = clean(Some(artist));
            }
        }
        names
    }

    /// Best effort: a failed write costs the title on the next play, not the download.
    fn save(&self, dir: &Path) {
        let mut text = String::new();
        if let Some(title) = &self.title {
            text.push_str(&format!("title\t{title}\n"));
        }
        if let Some(artist) = &self.artist {
            text.push_str(&format!("artist\t{artist}\n"));
        }
        let _ = std::fs::write(dir.join(NAMES_FILE), text);
    }
}

/// A file an earlier fetch left, when it is one this build can decode.
fn already_downloaded(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| extension(path).is_some_and(|ext| DECODABLE.contains(&ext.as_str())))
        .collect();
    found.sort();
    found.into_iter().next()
}

/// Refuse a container the pipeline cannot decode, by name, before it reaches the decoder.
pub fn check_decodable(path: &Path) -> Result<()> {
    let extension = extension(path).unwrap_or_default();
    anyhow::ensure!(
        DECODABLE.contains(&extension.as_str()),
        "yt-dlp wrote {}, which this build cannot decode (it reads {}). Install ffmpeg so \
         yt-dlp can convert to {AUDIO_FORMAT}.",
        path.display(),
        DECODABLE.join(", ")
    );
    anyhow::ensure!(
        std::fs::metadata(path)
            .map(|meta| meta.len() > 0)
            .unwrap_or(false),
        "yt-dlp wrote no audio to {}",
        path.display()
    );
    Ok(())
}

/// A metadata field, with yt-dlp's `NA` and blanks read as absent.
fn clean(field: Option<&str>) -> Option<String> {
    field
        .map(str::trim)
        .filter(|text| !text.is_empty() && *text != "NA")
        .map(str::to_string)
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use super::*;

    /// `FLACIFY_YTDLP` and `PATH` are process-wide, so the tests that set them take turns.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Poisoning is ignored on purpose: one failing test must not cascade into the rest,
    /// and each of these tests restores the variables it set before it asserts.
    fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn temp_root(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("flacify-youtube-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn given_a_youtube_url_when_parsed_then_the_video_id_is_extracted() {
        for url in [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=42",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
        ] {
            let request = parse(url).unwrap_or_else(|| panic!("{url} is a URL"));
            assert_eq!(request.video_id, "dQw4w9WgXcQ", "{url}");
        }
    }

    #[test]
    fn given_a_path_when_parsed_then_it_is_not_a_url() {
        for text in [
            "song.mp3",
            "./song.mp3",
            "/tmp/a file.flac",
            "ftp://x/y",
            "",
        ] {
            assert!(parse(text).is_none(), "{text} is not a URL");
        }
    }

    #[test]
    fn given_a_url_we_cannot_parse_when_parsed_then_it_still_gets_a_directory_name() {
        let request = parse("https://example.com/audio/1").expect("still a URL");
        assert!(request.video_id.starts_with("url-"), "{}", request.video_id);
        assert!(request
            .video_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-'));
    }

    #[test]
    fn given_a_hostile_id_when_parsed_then_it_cannot_leave_the_download_directory() {
        let request = parse("https://www.youtube.com/watch?v=../../etc/passwd").expect("a URL");
        assert!(!request.video_id.contains('/'), "{}", request.video_id);
        assert!(!request.video_id.contains(".."), "{}", request.video_id);
        let request = parse("https://youtu.be/..%2F..").expect("a URL");
        assert!(request.video_id.starts_with("url-"), "{}", request.video_id);
    }

    #[test]
    fn given_no_ytdlp_when_looking_for_it_then_nothing_is_found_and_the_error_says_how() {
        let _guard = env_lock();
        let empty = temp_root("no-ytdlp");
        std::fs::create_dir_all(&empty).expect("a temp dir");
        let path = std::env::var_os("PATH");
        let over = std::env::var_os(YTDLP_ENV);
        std::env::set_var("PATH", &empty);
        std::env::remove_var(YTDLP_ENV);

        let dir = temp_root("no-ytdlp-target");
        let request = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let error = fetch(&request, &dir, None).expect_err("no yt-dlp, no fetch");

        std::env::remove_var("PATH");
        if let Some(path) = path {
            std::env::set_var("PATH", path);
        }
        if let Some(over) = over {
            std::env::set_var(YTDLP_ENV, over);
        }
        let _ = std::fs::remove_dir_all(&empty);
        assert!(
            format!("{error:#}").contains("yt-dlp is not installed"),
            "{error:#}"
        );
    }

    #[test]
    fn given_a_fetch_when_run_then_the_printed_path_is_the_track() {
        let _guard = env_lock();
        let dir = temp_root("fetch");
        let script = fake_ytdlp(&dir, "prints a WAV");
        std::env::set_var(YTDLP_ENV, &script);

        let target = dir.join("downloads");
        let request = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let track = fetch(&request, &target, None).expect("the fake writes a WAV");
        std::env::remove_var(YTDLP_ENV);

        assert_eq!(extension(&track.path), Some("wav".to_string()));
        assert!(track.path.exists(), "{}", track.path.display());
        assert_eq!(track.title.as_deref(), Some("Some Title"));
        assert_eq!(track.artist.as_deref(), Some("Some Artist"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn given_a_second_fetch_of_the_same_id_when_run_then_no_child_is_spawned() {
        let _guard = env_lock();
        let dir = temp_root("reuse");
        let script = fake_ytdlp(&dir, "counts its starts");
        let count = dir.join("starts");
        std::env::set_var(YTDLP_ENV, &script);

        let target = dir.join("downloads");
        let request = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let first = fetch(&request, &target, None).expect("the first fetch downloads");
        let second = fetch(&request, &target, None).expect("the second reuses it");
        std::env::remove_var(YTDLP_ENV);

        assert_eq!(first.path, second.path);
        let starts = std::fs::read_to_string(&count).unwrap_or_default();
        assert_eq!(starts.lines().count(), 1, "yt-dlp ran {starts:?} times");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn given_a_download_this_build_cannot_decode_when_checked_then_it_is_refused_by_name() {
        let dir = temp_root("webm");
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let path = dir.join("audio.webm");
        std::fs::write(&path, b"not really a webm").expect("a file");
        let error = check_decodable(&path).expect_err("webm is not in the list");
        assert!(format!("{error:#}").contains("webm"), "{error:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn given_a_cancelled_token_when_fetching_then_it_is_cancelled_not_installed_missing() {
        let _guard = env_lock();
        let dir = temp_root("cancelled");
        let script = fake_ytdlp(&dir, "sleeps forever");
        std::env::set_var(YTDLP_ENV, &script);

        let cancel = Cancel::default();
        cancel.cancel();
        let target = dir.join("downloads");
        let request = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let error = fetch(&request, &target, Some(&cancel)).expect_err("cancelled");
        std::env::remove_var(YTDLP_ENV);

        assert!(process::is_cancelled(&error), "{error:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn given_a_download_then_a_replay_shows_the_title_without_asking_yt_dlp_again() {
        let _guard = env_lock();
        let dir = temp_root("names");
        let script = fake_ytdlp(&dir, "prints a WAV");
        std::env::set_var(YTDLP_ENV, &script);

        let target = dir.join("downloads");
        let request = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let first = fetch(&request, &target, None).expect("the first fetch downloads");
        // A yt-dlp that would fail if it ran again: the replay must not need it.
        std::env::set_var(YTDLP_ENV, dir.join("no-such-program"));
        let second = fetch(&request, &target, None).expect("the replay reuses the file");
        std::env::remove_var(YTDLP_ENV);

        assert_eq!(second.path, first.path);
        assert_eq!(second.title.as_deref(), Some("Some Title"));
        assert_eq!(second.artist.as_deref(), Some("Some Artist"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stand-in for yt-dlp: prints the fields the real one does, for a real WAV that a
    /// helper process writes on each call, so the client's contract is exercised with no
    /// network and no ffmpeg.
    fn fake_ytdlp(dir: &Path, mode: &str) -> PathBuf {
        std::fs::create_dir_all(dir).expect("a temp dir");
        let script = dir.join("fake-yt-dlp.sh");
        let template = dir.join("template.wav");
        write_wav(&template);
        let wav = dir.join("downloads/dQw4w9WgXcQ/audio.wav");
        std::fs::remove_file(&wav).ok();
        let body = format!(
            r#"#!/bin/sh
set -e
echo started >> "{dir}/starts"
if [ "{mode}" = "sleeps forever" ]; then sleep 30; fi
mkdir -p "$(dirname "{wav}")"
cp "{template}" "{wav}"
printf '%s\t%s\t%s\n' "{wav}" "Some Title" "Some Artist"
"#,
            dir = dir.display(),
            mode = mode,
            wav = wav.display(),
            template = template.display(),
        );
        std::fs::write(&script, body).expect("a fake script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(&script).expect("metadata").permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&script, permissions).expect("chmod");
        }
        script
    }

    /// A real one-second silent WAV, so a fetch's output is something the decoder accepts.
    fn write_wav(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("a target directory");
        }
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("a WAV");
        for _ in 0..48_000 {
            writer.write_sample(0i16).expect("a sample");
        }
        writer.finalize().expect("finalised");
    }
}
