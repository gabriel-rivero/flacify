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

/// Set (to anything) to ignore the copy under `FLACIFY_HOME`.
///
/// Only the tests set it: a developer machine has the app installed, so a test that means
/// "nothing is installed" must be able to say so without touching the real home. Nothing
/// in the app reads it.
const NO_MANAGED_ENV: &str = "FLACIFY_NO_MANAGED_TOOLS";

/// The container we ask for when ffmpeg is available. FLAC is in `symphonia`'s feature
/// list, is lossless so nothing is thrown away before the model sees it, and is written by
/// the `ffmpeg` yt-dlp already needs. The default (`bestaudio`) is WebM/Opus, which this
/// build cannot decode.
const AUDIO_FORMAT: &str = "flac";

/// The format we ask for when ffmpeg is absent: YouTube's own AAC in its own MP4, which
/// needs no postprocessing at all. `symphonia` decodes `mp4a` (AAC-LC) with the features
/// this build enables, verified by `flacify probe` on a real download (2026-11-28).
///
/// This is the difference between "a URL works after one 24 MB binary" and "a URL needs a
/// full ffmpeg install": yt-dlp only needs ffmpeg to remux or transcode, and asking it for
/// a format it can deliver untouched skips that step entirely. The cost is a lossy
/// source — which is what a lossy-streaming service holds anyway — instead of a FLAC that
/// was transcoded from it.
const NATIVE_FORMAT: &str = "140/bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio";

/// The containers the pipeline can actually decode, so a download it cannot read is named
/// here rather than failing later inside the decoder.
///
/// `ogg` is absent on purpose: the build enables the Ogg container and Vorbis but has no
/// Opus codec, so an Ogg file may pass an extension check and still fail to decode. Every
/// entry here is a container whose audio codec is always one this build has.
const DECODABLE: &[&str] = &["flac", "wav", "m4a", "mp3", "aac"];

/// The names a fetch printed, kept beside the audio so a replay shows the same title.
const NAMES_FILE: &str = "names.tsv";

/// Marks the one stdout line that carries the download's path, so it can be found among
/// the progress lines without guessing. Never shown to a user.
const SENTINEL: &str = "FLACIFY-FETCH";

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
    // A scheme is case-insensitive, and a URL without one is a path.
    let lower = url.get(..8).unwrap_or(url).to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
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
    // The fragment never reaches yt-dlp, so it must not change the id either: `#t=42`
    // and the bare URL are the same video and deserve the same download directory.
    let url = url.split('#').next().unwrap_or(url);
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let (authority, path) = after_scheme.split_once('/').unwrap_or((after_scheme, ""));
    // A userinfo section and a port are not the host, and hosts are case-insensitive.
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host).to_ascii_lowercase();
    let query = path.split_once('?').map(|(_, query)| query).unwrap_or("");
    let id = if host_is(&host, "youtu.be") {
        path.split(['?', '/']).next().unwrap_or("")
    } else if host_is(&host, "youtube.com") || host_is(&host, "youtube-nocookie.com") {
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

/// Whether `host` is `wanted` or a subdomain of it. A bare `ends_with` would let
/// `notyoutube.com` and `evilyoutu.be` through.
fn host_is(host: &str, wanted: &str) -> bool {
    host == wanted || host.ends_with(&format!(".{wanted}"))
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

/// Marks the one stdout line shape `enumerate` reads, the same trick `fetch` uses.
const LIST_SENTINEL: &str = "FLACIFY-LIST";

/// One track a playlist URL named, before any audio is fetched.
#[derive(Clone, Debug, PartialEq)]
pub struct Listed {
    pub request: Request,
    pub title: Option<String>,
    pub artist: Option<String>,
}

/// Whether a URL names a playlist rather than one video.
///
/// A `list=` parameter is a playlist unless it is a radio mix (`RD…`), which is only
/// expanded when the caller asks for radio: a share link carrying `start_radio=1` must
/// keep playing the single video it names, as it did before playlists existed.
pub fn is_playlist_url(text: &str) -> bool {
    let url = text.trim();
    if !has_scheme(url) {
        return false;
    }
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let (_, path) = after_scheme.split_once('/').unwrap_or((after_scheme, ""));
    let (path_only, rest) = match path.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (path, None),
    };
    let query = rest.map(|query| query.split('#').next().unwrap_or(query));
    let playlist_path = path_only.trim_end_matches('/').ends_with("playlist");
    let names_a_list = query
        .unwrap_or("")
        .split('&')
        .find_map(|pair| pair.strip_prefix("list="))
        .is_some_and(|value| !value.is_empty() && !value.starts_with("RD"));
    playlist_path || names_a_list
}

/// The seed video's radio mix: YouTube's `start_radio=1` recommendations.
pub fn radio_request(seed: &Request) -> Result<Request> {
    anyhow::ensure!(
        !seed.video_id.starts_with("url-"),
        "radio is a YouTube feature; {} is not a YouTube video",
        seed.url
    );
    let id = &seed.video_id;
    Ok(Request {
        url: format!("https://www.youtube.com/watch?v={id}&list=RD{id}&start_radio=1"),
        video_id: id.clone(),
    })
}

/// List a playlist or mix with `yt-dlp --flat-playlist`, metadata only. Nothing is
/// downloaded here; each entry is fetched by [`fetch`] when it is about to play.
pub fn enumerate(url: &str, cancel: Option<&Cancel>) -> Result<Vec<Listed>> {
    let program = find_ytdlp().context(
        "yt-dlp is not installed, and it is what lists a playlist. Install it with `brew \
         install yt-dlp`, `pipx install yt-dlp` or `pip install yt-dlp`, then try again.",
    )?;
    let fields =
        format!("{LIST_SENTINEL}\t%(id)s\t%(webpage_url,url)s\t%(title)s\t%(uploader,channel)s");
    let args = [
        "--flat-playlist",
        "--no-warnings",
        "--print",
        fields.as_str(),
        "--",
        url,
    ];
    let finished = process::run_streaming(&program, &args, cancel, &mut |_| {})
        .with_context(|| format!("listing {url}"))?;

    let youtube_source = is_youtube(url);
    let mut listed = Vec::new();
    for line in &finished.stdout {
        let Some(rest) = line.strip_prefix(LIST_SENTINEL) else {
            continue;
        };
        let rest = rest.trim_start_matches('\t');
        let mut parts = rest.splitn(4, '\t');
        let id = parts.next().unwrap_or("").trim();
        let webpage = parts.next().unwrap_or("").trim();
        let title = clean(parts.next());
        let artist = clean(parts.next());
        let request = if has_scheme(webpage) {
            let video_id =
                video_id_of(webpage).unwrap_or_else(|| format!("url-{}", short_hash(webpage)));
            Some(Request {
                url: webpage.to_string(),
                video_id,
            })
        } else if youtube_source && sanitised(id).is_some() {
            Some(Request {
                url: format!("https://www.youtube.com/watch?v={id}"),
                video_id: id.to_string(),
            })
        } else {
            None
        };
        if let Some(request) = request {
            listed.push(Listed {
                request,
                title,
                artist,
            });
        }
    }
    anyhow::ensure!(!listed.is_empty(), "yt-dlp listed no tracks for {url}");
    Ok(listed)
}

fn has_scheme(text: &str) -> bool {
    let text = text.get(..8).unwrap_or(text).to_ascii_lowercase();
    text.starts_with("http://") || text.starts_with("https://")
}

fn is_youtube(url: &str) -> bool {
    let url = url.split('#').next().unwrap_or(url);
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let (authority, _) = after_scheme.split_once('/').unwrap_or((after_scheme, ""));
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host).to_ascii_lowercase();
    host_is(&host, "youtube.com")
        || host_is(&host, "youtu.be")
        || host_is(&host, "youtube-nocookie.com")
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

/// Where `yt-dlp` is: the override, then the copy this app installed, then `PATH`.
///
/// The managed copy comes before `PATH` because it is the one `flacify setup` just put
/// there: a machine with a broken older yt-dlp on `PATH` would otherwise keep using it.
/// `None` means the error is worth raising, not that a download should be skipped.
pub fn find_ytdlp() -> Option<PathBuf> {
    let managed = crate::setup::Paths::discover()
        .ok()
        .map(|paths| paths.managed_ytdlp());
    find_ytdlp_with(managed.as_deref())
}

/// [`find_ytdlp`] with the managed path supplied.
///
/// The managed copy lives under `FLACIFY_HOME`, so a lookup inside a test must not fall
/// back to the real one: a machine with the app installed has a working `yt-dlp` in
/// `~/AppData/Local/flacify/runtime/bin`, and a test that only emptied `PATH` would find
/// it, reach the network and download a real video. `FLACIFY_HOME` pointed at a temp
/// directory is what keeps that from happening, and `FLACIFY_NO_MANAGED_TOOLS` turns the
/// managed lookup off entirely for a test that must see "nothing installed".
fn find_ytdlp_with(managed: Option<&Path>) -> Option<PathBuf> {
    if let Some(program) = std::env::var_os(YTDLP_ENV) {
        let path = PathBuf::from(program);
        if path.components().count() > 1 {
            return process::is_executable(&path).then_some(path);
        }
        return process::find_on_path(&path.to_string_lossy());
    }
    if std::env::var_os(NO_MANAGED_ENV).is_none() {
        if let Some(managed) = managed {
            if process::is_executable(managed) {
                return Some(managed.to_path_buf());
            }
        }
    }
    process::find_on_path("yt-dlp")
}

/// The yt-dlp arguments for one fetch, which depend on whether ffmpeg can be found.
///
/// With ffmpeg: ask for FLAC, which is lossless and is what `symphonia` decodes; yt-dlp
/// transcodes the source into it. Without ffmpeg: ask for a format YouTube already serves
/// as a decodable container, so no postprocessor runs. Asking for FLAC without ffmpeg is
/// the exact failure this avoids — yt-dlp downloads the audio and then dies at the
/// conversion step with "ffprobe and ffmpeg not found", having thrown the bytes away.
fn fetch_args<'a>(template: &'a str, fields: &'a str, url: &'a str) -> Vec<&'a str> {
    fetch_args_with(template, fields, url, has_ffmpeg())
}

/// The same arguments with the ffmpeg decision supplied, so both branches are testable
/// without a machine that has (or lacks) ffmpeg.
fn fetch_args_with<'a>(
    template: &'a str,
    fields: &'a str,
    url: &'a str,
    ffmpeg: bool,
) -> Vec<&'a str> {
    let mut args = vec![
        "--no-playlist",
        "--no-warnings",
        "--no-continue",
        "--newline",
        "-o",
        template,
        "--print",
        fields,
        url,
    ];
    if ffmpeg {
        args.splice(
            4..4,
            ["-x", "--audio-format", AUDIO_FORMAT, "--audio-quality", "0"],
        );
    } else {
        args.splice(4..4, ["-f", NATIVE_FORMAT]);
    }
    args
}

/// Whether an ffmpeg any yt-dlp will find is on `PATH`.
///
/// Not a `--version` spawn: yt-dlp does its own lookup, and all this decides is which
/// format to ask for. A wrong answer here costs a re-fetch, never a wrong file — the
/// fetched file is probed by [`check_decodable`] before anything trusts it.
pub fn has_ffmpeg() -> bool {
    process::find_on_path("ffmpeg").is_some()
}

/// The file a fetch of `request` produced, or the one an earlier fetch already left.
///
/// A file already in `dir` is trusted: it is keyed by the video id, so another video
/// cannot have written it, and `flacify cache --evict` is how it is reclaimed or
/// refreshed. The second play therefore spawns nothing, and shows the same title because
/// the names are written beside the audio rather than asked for again.
pub fn fetch(request: &Request, dir: &Path, cancel: Option<&Cancel>) -> Result<Track> {
    let target = dir.join(&request.video_id);
    if let Some(existing) = already_downloaded(&target, &request.video_id) {
        // A zero-length or truncated file left by an interrupted run must not be trusted
        // forever: every replay would fail in the decoder and nothing would re-fetch.
        if check_decodable(&existing).is_ok() {
            let names = Names::load(&target);
            return Ok(Track {
                path: existing,
                title: names.title,
                artist: names.artist,
            });
        }
        tracing::warn!(
            file = %existing.display(),
            "the stored download is unreadable; fetching it again"
        );
        let _ = std::fs::remove_file(&existing);
    }

    let program = find_ytdlp().context(
        "yt-dlp is not installed, and it is what fetches a URL. Install it with `brew install \
         yt-dlp`, `pipx install yt-dlp` or `pip install yt-dlp`, then try again.",
    )?;
    std::fs::create_dir_all(&target).with_context(|| format!("creating {}", target.display()))?;

    let template = target.join("%(id)s.%(ext)s");
    // The sentinel is what makes the printed line findable: without it, every progress
    // line also "looks like" a path, since it has no tab. A tab is a legal byte in a
    // title, so the fields are separated by one only because the sentinel is first.
    let fields =
        format!("after_move:{SENTINEL}\t%(filepath)s\t%(title)s\t%(artist,uploader,channel)s");
    let template = template.to_string_lossy().into_owned();
    let args = fetch_args(&template, &fields, &request.url);
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
        .find_map(|line| {
            line.strip_prefix(SENTINEL)
                .map(|rest| rest.trim_start_matches('\t'))
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
            let Some((field, value)) = line.split_once('\t') else {
                continue;
            };
            let value = unescape(value);
            match field {
                "title" => names.title = Some(value),
                "artist" => names.artist = Some(value),
                _ => {}
            }
        }
        names
    }

    /// Best effort: a failed write costs the title on the next play, not the download.
    fn save(&self, dir: &Path) {
        let mut text = String::new();
        if let Some(title) = &self.title {
            text.push_str(&format!("title\t{}\n", escape(title)));
        }
        if let Some(artist) = &self.artist {
            text.push_str(&format!("artist\t{}\n", escape(artist)));
        }
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(NAMES_FILE), text);
    }
}

/// A title is arbitrary text and the file is line- and tab-delimited, so both have to be
/// escaped: a title containing a newline would otherwise truncate, and one containing
/// `\nartist\tX` would forge a line.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// A file an earlier fetch left, when it is one this build can decode.
///
/// The file named after the id wins over any other decodable file in the directory.
/// `flacify enhance <url>` writes its output beside the source (`<name>.flacify.wav`),
/// which is also decodable; picking by name order alone would make `play` on that id
/// quietly return the render instead of the fetch.
fn already_downloaded(dir: &Path, video_id: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| extension(path).is_some_and(|ext| DECODABLE.contains(&ext.as_str())))
        .collect();
    found.sort();
    found
        .iter()
        .find(|path| path.file_stem().is_some_and(|stem| stem == video_id))
        .or_else(|| found.first())
        .cloned()
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
    // The extension is a cheap filter, not proof: a truncated file, or a right container
    // holding a codec this build lacks, passes it. Probing reads headers only, and this
    // runs once per fetch rather than per window.
    crate::decode::probe_headers(path)?;
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
    fn given_a_fragment_or_uppercase_host_when_parsed_then_the_same_id_comes_back() {
        // A fragment never reaches yt-dlp and a host is case-insensitive, so neither may
        // change the id: it is the download directory's name.
        for url in [
            "https://youtu.be/dQw4w9WgXcQ#t=42",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ#t=42",
            "HTTPS://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
            "https://WWW.YouTube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com:443/watch?v=dQw4w9WgXcQ",
        ] {
            let request = parse(url).unwrap_or_else(|| panic!("{url} is a URL"));
            assert_eq!(request.video_id, "dQw4w9WgXcQ", "{url}");
        }
    }

    #[test]
    fn given_a_lookalike_host_when_parsed_then_it_is_keyed_by_its_url_not_a_youtube_id() {
        for url in [
            "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
            "https://evilyoutu.be/dQw4w9WgXcQ",
        ] {
            let request = parse(url).unwrap_or_else(|| panic!("{url} is a URL"));
            assert!(
                request.video_id.starts_with("url-"),
                "{url} gave {}",
                request.video_id
            );
        }
    }

    #[test]
    fn given_a_title_with_a_tab_or_newline_when_saved_and_loaded_then_it_round_trips() {
        let dir = temp_root("names-escape");
        let names = Names {
            title: Some("a\tb\nc\\d".to_string()),
            artist: Some("x\ty".to_string()),
        };
        names.save(&dir);
        let loaded = Names::load(&dir);
        assert_eq!(loaded.title.as_deref(), Some("a\tb\nc\\d"));
        assert_eq!(loaded.artist.as_deref(), Some("x\ty"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn given_a_fetch_when_ffmpeg_is_present_then_it_asks_for_a_transcode_to_flac() {
        // The FLAC request only works with a postprocessor, so the two must move together:
        // asking for FLAC without ffmpeg is the "ffprobe and ffmpeg not found" failure.
        let args = fetch_args_with("out.%(ext)s", "fields", "https://youtu.be/x", true);
        assert!(args.contains(&"-x"), "{args:?}");
        assert!(args.contains(&"--audio-format"), "{args:?}");
        assert!(args.contains(&AUDIO_FORMAT), "{args:?}");
        assert!(
            !args.contains(&"-f"),
            "a transcode does not need an explicit format: {args:?}"
        );
    }

    #[test]
    fn given_a_fetch_without_ffmpeg_then_it_asks_for_a_format_that_needs_no_postprocessing() {
        // No `-x` and no `--audio-format`: yt-dlp writes the container as served, which is
        // the only way a download succeeds with no ffmpeg on the machine.
        let args = fetch_args_with("out.%(ext)s", "fields", "https://youtu.be/x", false);
        assert!(!args.contains(&"-x"), "{args:?}");
        assert!(!args.contains(&"--audio-format"), "{args:?}");
        assert!(args.contains(&"-f"), "{args:?}");
        assert!(args.contains(&NATIVE_FORMAT), "{args:?}");
    }

    #[test]
    fn given_the_native_format_choices_then_each_names_a_container_we_decode() {
        // The string falls back in order; every pick must be a container `DECODABLE`
        // knows, or the audio is downloaded and then thrown away by the decoder.
        for pick in NATIVE_FORMAT.split('/') {
            let ext = pick
                .rsplit_once("ext=")
                .map(|(_, ext)| ext.trim_end_matches(']'))
                // A bare format id (`140`) is the m4a one documented in `NATIVE_FORMAT`.
                .unwrap_or("m4a");
            assert!(
                DECODABLE.contains(&ext),
                "{pick:?} yields {ext:?}, which this build cannot decode"
            );
        }
    }

    #[test]
    fn given_a_title_that_looks_like_a_field_when_loaded_then_it_cannot_forge_one() {
        let dir = temp_root("names-forge");
        std::fs::create_dir_all(&dir).expect("a temp dir");
        std::fs::write(dir.join(NAMES_FILE), "title\tinnocent\\nartist\\tforged\n")
            .expect("a names file");
        let loaded = Names::load(&dir);
        assert_eq!(loaded.title.as_deref(), Some("innocent\nartist\tforged"));
        assert_eq!(loaded.artist, None);
        let _ = std::fs::remove_dir_all(&dir);
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
    fn given_no_ytdlp_anywhere_when_looking_for_it_then_nothing_is_found() {
        let _guard = env_lock();
        let empty = temp_root("no-ytdlp");
        std::fs::create_dir_all(&empty).expect("a temp dir");
        let path = std::env::var_os("PATH");
        let over = std::env::var_os(YTDLP_ENV);
        let managed_env = std::env::var_os(NO_MANAGED_ENV);
        std::env::set_var("PATH", &empty);
        std::env::remove_var(YTDLP_ENV);
        // A developer machine has the app installed, so its real `runtime/bin/yt-dlp` would
        // be found here and a fetch would reach the network. Say "nothing installed".
        std::env::set_var(NO_MANAGED_ENV, "1");

        let dir = temp_root("no-ytdlp-target");
        let request = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let error = fetch(&request, &dir, None).expect_err("no yt-dlp, no fetch");

        std::env::remove_var("PATH");
        std::env::remove_var(NO_MANAGED_ENV);
        if let Some(path) = path {
            std::env::set_var("PATH", path);
        }
        if let Some(over) = over {
            std::env::set_var(YTDLP_ENV, over);
        }
        if let Some(managed) = managed_env {
            std::env::set_var(NO_MANAGED_ENV, managed);
        }
        let _ = std::fs::remove_dir_all(&empty);
        assert!(
            format!("{error:#}").contains("yt-dlp is not installed"),
            "{error:#}"
        );
    }

    // The fake yt-dlp is a `#!/bin/sh` script, so it does not run on Windows.
    #[cfg(unix)]
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

    // The fake yt-dlp is a `#!/bin/sh` script, so it does not run on Windows.
    #[cfg(unix)]
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

    // The fake yt-dlp is a `#!/bin/sh` script, so it does not run on Windows.
    #[cfg(unix)]
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

    // The fake yt-dlp is a `#!/bin/sh` script, so it does not run on Windows.
    #[cfg(unix)]
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

    #[test]
    fn given_urls_when_asked_if_they_are_playlists_then_radio_mixes_are_not() {
        for url in [
            "https://www.youtube.com/playlist?list=PL123",
            "https://www.youtube.com/watch?v=abc&list=PL123",
            "https://www.youtube.com/playlist",
        ] {
            assert!(is_playlist_url(url), "{url} is a playlist");
        }
        for url in [
            "https://youtu.be/abc",
            "https://www.youtube.com/watch?v=abc",
            "https://www.youtube.com/watch?v=abc&list=RDabc&start_radio=1",
            "song.mp3",
        ] {
            assert!(!is_playlist_url(url), "{url} is not a playlist");
        }
    }

    #[test]
    fn given_a_seed_video_when_made_radio_then_the_url_asks_for_its_mix() {
        let seed = parse("https://youtu.be/dQw4w9WgXcQ").expect("a URL");
        let radio = radio_request(&seed).expect("YouTube does radio");
        assert!(radio.url.contains("v=dQw4w9WgXcQ"), "{}", radio.url);
        assert!(radio.url.contains("list=RDdQw4w9WgXcQ"), "{}", radio.url);
        assert!(radio.url.contains("start_radio=1"), "{}", radio.url);

        let other = parse("https://example.com/audio/1").expect("a URL");
        let error = radio_request(&other).expect_err("radio is YouTube-only");
        assert!(format!("{error:#}").contains("YouTube"), "{error:#}");
    }

    // The fake yt-dlp is a `#!/bin/sh` script, so it does not run on Windows.
    #[cfg(unix)]
    #[test]
    fn given_a_playlist_when_enumerated_then_each_printed_line_becomes_an_entry() {
        let _guard = env_lock();
        let dir = temp_root("enumerate");
        let script = fake_listing(
            &dir,
            "FLACIFY-LIST\tone\thttps://www.youtube.com/watch?v=one\tFirst\tArtist A\n\
             FLACIFY-LIST\ttwo\tNA\tSecond\tNA\n",
        );
        std::env::set_var(YTDLP_ENV, &script);

        let listed = enumerate("https://www.youtube.com/playlist?list=PL1", None)
            .expect("the fake lists two");
        std::env::remove_var(YTDLP_ENV);

        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].request.video_id, "one");
        assert_eq!(listed[0].title.as_deref(), Some("First"));
        assert_eq!(listed[0].artist.as_deref(), Some("Artist A"));
        // No webpage URL, but the source is YouTube, so the id is enough to fetch it.
        assert_eq!(listed[1].request.url, "https://www.youtube.com/watch?v=two");
        assert_eq!(listed[1].title.as_deref(), Some("Second"));
        assert_eq!(listed[1].artist, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn given_no_ytdlp_when_listing_a_playlist_then_the_error_says_how_to_install_it() {
        let _guard = env_lock();
        let empty = temp_root("no-ytdlp-list");
        std::fs::create_dir_all(&empty).expect("a temp dir");
        let path = std::env::var_os("PATH");
        let over = std::env::var_os(YTDLP_ENV);
        let managed_env = std::env::var_os(NO_MANAGED_ENV);
        std::env::set_var("PATH", &empty);
        std::env::remove_var(YTDLP_ENV);
        std::env::set_var(NO_MANAGED_ENV, "1");

        let error = enumerate("https://www.youtube.com/playlist?list=PL1", None)
            .expect_err("no yt-dlp, no listing");

        std::env::remove_var("PATH");
        std::env::remove_var(NO_MANAGED_ENV);
        if let Some(path) = path {
            std::env::set_var("PATH", path);
        }
        if let Some(over) = over {
            std::env::set_var(YTDLP_ENV, over);
        }
        if let Some(managed) = managed_env {
            std::env::set_var(NO_MANAGED_ENV, managed);
        }
        let _ = std::fs::remove_dir_all(&empty);
        assert!(
            format!("{error:#}").contains("yt-dlp is not installed"),
            "{error:#}"
        );
    }

    /// A stand-in for yt-dlp that prints the listing lines given, ignoring its arguments.
    #[cfg(unix)]
    fn fake_listing(dir: &Path, lines: &str) -> PathBuf {
        std::fs::create_dir_all(dir).expect("a temp dir");
        let script = dir.join("fake-list-yt-dlp.sh");
        let body = format!(
            "#!/bin/sh\nprintf '%s' '{}'\n",
            lines.replace('\'', r"'\''")
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

    /// A stand-in for yt-dlp: prints the fields the real one does, for a real WAV that a
    /// helper process writes on each call, so the client's contract is exercised with no
    /// network and no ffmpeg.
    #[cfg(unix)]
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
printf '%s\t%s\t%s\t%s\n' "FLACIFY-FETCH" "{wav}" "Some Title" "Some Artist"
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
    #[cfg(unix)]
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
