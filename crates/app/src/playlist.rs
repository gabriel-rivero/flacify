//! The queue: what plays after this, and where each track comes from.
//!
//! Local inputs (folder, `.m3u`, `.pls`) expand now; a remote playlist is listed with
//! `yt-dlp --flat-playlist` and each track is fetched only when it is about to play.
//! No device, decoder or window here: the window advances the cursor and [`resolve`]
//! gives the pipeline a path.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::batch;
use crate::process::Cancel;
use crate::youtube::{self, Request};

/// One thing the queue can play, before any audio exists for it.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Local { path: PathBuf, label: String },
    Remote { request: Request, label: String },
}

impl Entry {
    fn local(path: PathBuf) -> Self {
        let label = file_name(&path);
        Entry::Local { path, label }
    }

    fn remote(request: Request, title: Option<String>) -> Self {
        let label = title.unwrap_or_else(|| request.video_id.clone());
        Entry::Remote { request, label }
    }

    fn from_listed(listed: youtube::Listed) -> Self {
        Entry::remote(listed.request, listed.title)
    }

    /// The name the queue strip shows.
    pub fn label(&self) -> &str {
        match self {
            Entry::Local { label, .. } | Entry::Remote { label, .. } => label,
        }
    }

    /// The file this entry plays, when it already is one.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            Entry::Local { path, .. } => Some(path),
            Entry::Remote { .. } => None,
        }
    }
}

/// An entry that has become a file the pipeline can open.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub path: PathBuf,
    pub title: Option<String>,
    pub artist: Option<String>,
}

impl Resolved {
    pub fn label(&self) -> String {
        self.title.clone().unwrap_or_else(|| file_name(&self.path))
    }
}

/// A list of entries and the cursor into it.
#[derive(Debug)]
pub struct Queue {
    entries: Vec<Entry>,
    cursor: usize,
    downloads: PathBuf,
    origin: String,
}

impl Queue {
    /// Expand local inputs into a queue: a folder walks its audio files, an `.m3u`/`.m3u8`
    /// or `.pls` is parsed, and anything else is a single track.
    pub fn from_local(inputs: &[PathBuf], downloads: &Path) -> Result<Queue> {
        anyhow::ensure!(!inputs.is_empty(), "no input was given");
        let mut entries = Vec::new();
        let mut origin = String::new();
        for input in inputs {
            if input.is_dir() {
                origin = format!("folder · {}", input.display());
                for path in batch::audio_files(input) {
                    entries.push(Entry::local(path));
                }
                continue;
            }
            let extension = extension(input).unwrap_or_default();
            match extension.as_str() {
                "m3u" | "m3u8" => {
                    let base = input.parent().unwrap_or_else(|| Path::new("."));
                    let text = std::fs::read_to_string(input)
                        .with_context(|| format!("reading {}", input.display()))?;
                    origin = format!("playlist · {}", input.display());
                    push_local(&mut entries, parse_m3u(&text, base));
                }
                "pls" => {
                    let base = input.parent().unwrap_or_else(|| Path::new("."));
                    let text = std::fs::read_to_string(input)
                        .with_context(|| format!("reading {}", input.display()))?;
                    origin = format!("playlist · {}", input.display());
                    push_local(&mut entries, parse_pls(&text, base));
                }
                _ => {
                    anyhow::ensure!(input.is_file(), "{} is not a file", input.display());
                    if origin.is_empty() {
                        origin = format!("file · {}", input.display());
                    }
                    entries.push(Entry::local(input.clone()));
                }
            }
        }
        anyhow::ensure!(
            !entries.is_empty(),
            "nothing playable in {}",
            inputs
                .iter()
                .map(|input| input.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(Queue {
            entries,
            cursor: 0,
            downloads: downloads.to_path_buf(),
            origin,
        })
    }

    /// A URL becomes a queue: `--radio` expands a video into its mix, a playlist URL is
    /// enumerated, and anything else is one track, which is the old behaviour.
    pub fn from_url(url: &str, downloads: &Path, radio: bool, single: bool) -> Result<Queue> {
        let text = url.trim();
        let request = youtube::parse(text).with_context(|| format!("{text} is not a URL"))?;
        let entries = if single {
            vec![Entry::remote(request, None)]
        } else if radio {
            let seed = youtube::radio_request(&request)?;
            youtube::enumerate(&seed.url, None)?
                .into_iter()
                .map(Entry::from_listed)
                .collect()
        } else if youtube::is_playlist_url(text) {
            youtube::enumerate(text, None)?
                .into_iter()
                .map(Entry::from_listed)
                .collect()
        } else {
            vec![Entry::remote(request, None)]
        };
        anyhow::ensure!(!entries.is_empty(), "yt-dlp listed no tracks for {text}");
        Ok(Queue {
            entries,
            cursor: 0,
            downloads: downloads.to_path_buf(),
            origin: format!("url · {text}"),
        })
    }

    pub fn downloads(&self) -> &Path {
        &self.downloads
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn current(&self) -> Option<&Entry> {
        self.entries.get(self.cursor)
    }

    pub fn at(&self, index: usize) -> Option<&Entry> {
        self.entries.get(index)
    }

    pub fn next_index(&self) -> Option<usize> {
        (self.cursor + 1 < self.entries.len()).then_some(self.cursor + 1)
    }

    pub fn previous_index(&self) -> Option<usize> {
        self.cursor.checked_sub(1)
    }

    /// Move to the next entry. `false` at the end, where the cursor stays put.
    pub fn advance(&mut self) -> bool {
        match self.next_index() {
            Some(next) => {
                self.cursor = next;
                true
            }
            None => false,
        }
    }

    /// Move to the previous entry. `false` at the start.
    pub fn rewind(&mut self) -> bool {
        match self.previous_index() {
            Some(previous) => {
                self.cursor = previous;
                true
            }
            None => false,
        }
    }

    /// "3 / 12" for the queue strip.
    pub fn indicator(&self) -> String {
        format!("{} / {}", self.cursor + 1, self.entries.len())
    }
}

/// Turn the current entry into a file: local entries already are one, remote entries are
/// fetched (and reused, exactly as a single `play <url>` does).
pub fn resolve(entry: &Entry, downloads: &Path, cancel: Option<&Cancel>) -> Result<Resolved> {
    match entry {
        Entry::Local { path, .. } => Ok(Resolved {
            path: path.clone(),
            title: None,
            artist: None,
        }),
        Entry::Remote { request, .. } => {
            let track = youtube::fetch(request, downloads, cancel)?;
            Ok(Resolved {
                path: track.path,
                title: track.title,
                artist: track.artist,
            })
        }
    }
}

/// A track a playlist named, before it is trusted as a file.
struct LocalItem {
    path: PathBuf,
    title: Option<String>,
}

fn push_local(entries: &mut Vec<Entry>, items: Vec<LocalItem>) {
    for item in items {
        if !item.path.is_file() {
            tracing::warn!(path = %item.path.display(), "playlist entry is not a file; skipping");
            continue;
        }
        entries.push(Entry::Local {
            label: item.title.clone().unwrap_or_else(|| file_name(&item.path)),
            path: item.path,
        });
    }
}

/// Parse an M3U/M3U8 playlist. `#EXTINF:` carries `duration,title`; every other `#` line
/// is a directive. Relative paths resolve against the playlist's directory.
fn parse_m3u(text: &str, base: &Path) -> Vec<LocalItem> {
    let mut items = Vec::new();
    let mut pending_title: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("#EXTINF:") {
            pending_title = rest
                .split_once(',')
                .map(|(_, title)| title.trim())
                .filter(|title| !title.is_empty())
                .map(str::to_string);
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        match local_path(line, base) {
            Some(path) => items.push(LocalItem {
                path,
                title: pending_title.take(),
            }),
            None => pending_title = None,
        }
    }
    items
}

/// Parse a PLS playlist, in index order. `FileN=` names a track and `TitleN=` names it.
fn parse_pls(text: &str, base: &Path) -> Vec<LocalItem> {
    let mut rows: BTreeMap<u32, (Option<String>, Option<String>)> = BTreeMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('[') || line.starts_with(';') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let (field, number) = split_field(&key);
        let Ok(index) = number.parse::<u32>() else {
            continue;
        };
        let value = value.trim();
        let row = rows.entry(index).or_default();
        match field {
            "file" => {
                if !is_http(value) && !value.is_empty() {
                    row.0 = Some(value.to_string());
                }
            }
            "title" if !value.is_empty() => row.1 = Some(value.to_string()),
            _ => {}
        }
    }
    rows.into_values()
        .filter_map(|(file, title)| {
            let file = file?;
            local_path(&file, base).map(|path| LocalItem { path, title })
        })
        .collect()
}

/// Split a PLS key into its letters and its trailing digits ("File2" -> ("file", "2")).
fn split_field(key: &str) -> (&str, &str) {
    let at = key
        .find(|character: char| character.is_ascii_digit())
        .unwrap_or(key.len());
    key.split_at(at)
}

/// A playlist line as a path, or `None` for a URL or a blank.
fn local_path(text: &str, base: &Path) -> Option<PathBuf> {
    let text = text.trim().trim_matches('"');
    if text.is_empty() || is_http(text) {
        return None;
    }
    let path = PathBuf::from(text);
    Some(if path.is_absolute() {
        path
    } else {
        base.join(path)
    })
}

fn is_http(text: &str) -> bool {
    let text = text.get(..8).unwrap_or(text).to_ascii_lowercase();
    text.starts_with("http://") || text.starts_with("https://")
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "flacify-playlist-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("a temp dir");
        root
    }

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("a parent");
        }
        std::fs::write(path, []).expect("an empty file");
    }

    #[test]
    fn given_an_m3u_with_comments_titles_and_relative_paths_when_parsed_then_the_tracks_are_in_order(
    ) {
        let base = Path::new("/music");
        let text = "#EXTM3U\n#EXTINF:123,First Song\none.mp3\n# a comment\n\nsecond.flac\n";
        let items = parse_m3u(text, base);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].path, Path::new("/music/one.mp3"));
        assert_eq!(items[0].title.as_deref(), Some("First Song"));
        assert_eq!(items[1].path, Path::new("/music/second.flac"));
        assert_eq!(items[1].title, None);
    }

    #[test]
    fn given_absolute_and_url_entries_when_parsed_then_paths_are_kept_and_urls_are_not() {
        let base = Path::new("/music");
        let items = parse_m3u("/elsewhere/a.flac\nhttps://example.com/b.mp3\n", base);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, Path::new("/elsewhere/a.flac"));
    }

    #[test]
    fn given_a_pls_out_of_order_when_parsed_then_it_is_sorted_by_index() {
        let base = Path::new("/music");
        let text = "[playlist]\nNumberOfEntries=2\nTitle2=Second\nFile2=b.mp3\nFile1=a.mp3\nTitle1=First\n";
        let items = parse_pls(text, base);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].path, Path::new("/music/a.mp3"));
        assert_eq!(items[0].title.as_deref(), Some("First"));
        assert_eq!(items[1].path, Path::new("/music/b.mp3"));
    }

    #[test]
    fn given_a_folder_when_opened_as_a_queue_then_audio_files_are_sorted_and_others_ignored() {
        let root = temp_root("folder");
        touch(&root.join("b.mp3"));
        touch(&root.join("a.flac"));
        touch(&root.join("notes.txt"));
        let queue = Queue::from_local(std::slice::from_ref(&root), &root).expect("a folder queue");
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.at(0).unwrap().label(), "a.flac");
        assert_eq!(queue.at(1).unwrap().label(), "b.mp3");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_playlist_naming_missing_files_when_opened_then_only_existing_files_are_queued() {
        let root = temp_root("missing");
        touch(&root.join("real.mp3"));
        let list = root.join("list.m3u");
        std::fs::write(&list, "real.mp3\nmissing.mp3\n").expect("a playlist");
        let queue = Queue::from_local(&[list], &root).expect("a playlist queue");
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.current().unwrap().label(), "real.mp3");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_empty_playlist_when_opened_then_it_is_an_error_that_names_the_input() {
        let root = temp_root("empty");
        let list = root.join("list.m3u");
        std::fs::write(&list, "# nothing here\n").expect("a playlist");
        let error = Queue::from_local(&[list], &root).expect_err("nothing to play");
        assert!(
            format!("{error:#}").contains("nothing playable"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_queue_when_advanced_and_rewound_then_it_stops_at_the_ends() {
        let root = temp_root("cursor");
        touch(&root.join("a.mp3"));
        touch(&root.join("b.mp3"));
        let mut queue = Queue::from_local(std::slice::from_ref(&root), &root).expect("a queue");
        assert_eq!(queue.indicator(), "1 / 2");
        assert!(!queue.rewind());
        assert!(queue.advance());
        assert_eq!(queue.indicator(), "2 / 2");
        assert!(!queue.advance());
        assert!(queue.rewind());
        assert_eq!(queue.cursor(), 0);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_remote_playlist_entry_when_queued_then_it_resolves_through_ytdlp_not_a_file() {
        let root = temp_root("remote");
        let queue = Queue::from_url("https://youtu.be/dQw4w9WgXcQ", &root, false, true)
            .expect("one video is a queue of one");
        assert_eq!(queue.len(), 1);
        assert!(queue.current().unwrap().local_path().is_none());
        assert_eq!(queue.current().unwrap().label(), "dQw4w9WgXcQ");
        std::fs::remove_dir_all(root).ok();
    }
}
