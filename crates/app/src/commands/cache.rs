//! `flacify cache`: what the cache holds, and evicting down to the cap.
//!
//! The plan is printed before anything is removed, so the same run both says what it is
//! about to do and does it. Accounting alone is the default: `--evict` is the only flag
//! that deletes.

use anyhow::Result;

use crate::cache::{
    human_bytes, now, Cache, Eviction, EvictionPlan, Removal, Usage, DEFAULT_CAP_BYTES,
};
use crate::setup::Paths;

pub(crate) fn run(evict: bool, cap_gb: Option<f64>) -> Result<()> {
    let paths = Paths::discover()?;
    let cache = Cache::new(paths.cache_dir());
    let cap = cap_bytes(cap_gb)?;

    print_usage(&cache.usage(), cap);
    // Fetched audio lives outside the cache, so the cache total would quietly omit it.
    // Reported here rather than evicted: a re-download is the user's call, and deleting a
    // 10-minute video to reclaim space should not be a side effect of a cache command.
    println!(
        "  {:<10} {:>11} of fetched audio",
        "downloads",
        human_bytes(crate::setup::dir_size(&paths.downloads_dir()))
    );
    if !evict {
        return Ok(());
    }

    let plan = cache.eviction_plan(cap);
    print_plan(&plan);
    if plan.sweeps.is_empty() && plan.tiers.is_empty() && plan.entries.is_empty() {
        println!("\nnothing to evict");
        return Ok(());
    }
    let outcome = cache.evict(&plan)?;
    print_outcome(&outcome, &plan);
    Ok(())
}

/// The cap in bytes: `--cap-gb`, or the 20 GB default. A fraction is allowed so a test can
/// aim at a cache it can build in a second.
fn cap_bytes(cap_gb: Option<f64>) -> Result<u64> {
    let Some(gb) = cap_gb else {
        return Ok(DEFAULT_CAP_BYTES);
    };
    anyhow::ensure!(
        gb.is_finite() && gb >= 0.0,
        "--cap-gb needs a size in gigabytes, not {gb}"
    );
    Ok((gb * 1e9).round() as u64)
}

fn print_usage(usage: &Usage, cap: u64) {
    println!("cache    {}", crate::cache::CACHE_VERSION);
    tally("rendered", usage.rendered.entries, usage.rendered.bytes);
    tally("partial", usage.partial.entries, usage.partial.bytes);
    tally("source", usage.source.entries, usage.source.bytes);
    tally("unknown", usage.unknown.entries, usage.unknown.bytes);
    tally("stale", usage.stale.entries, usage.stale.bytes);
    let total = usage.total();
    println!(
        "  {:<10} {:>11} of a {} cap ({:.0}%)",
        "total",
        human_bytes(total),
        human_bytes(cap),
        percent(total, cap)
    );
    if usage.unaccounted_bytes > 0 {
        println!(
            "  {:<10} {:>11} on disk that the manifests do not name",
            "note",
            human_bytes(usage.unaccounted_bytes)
        );
    }
    if usage.overstated_entries > 0 {
        println!(
            "  note       {} whose manifest names more than its files hold",
            plural(usage.overstated_entries, "entry", "entries")
        );
    }
}

fn tally(kind: &str, entries: usize, bytes: u64) {
    println!(
        "  {kind:<10} {:>12}  {:>11}",
        plural(entries, "entry", "entries"),
        human_bytes(bytes)
    );
}

fn print_plan(plan: &EvictionPlan) {
    let excess = plan.before.saturating_sub(plan.cap);
    println!(
        "\nplan     {} over the {} cap, {} -> {}",
        human_bytes(excess),
        human_bytes(plan.cap),
        human_bytes(plan.before),
        human_bytes(plan.after)
    );
    for removal in &plan.sweeps {
        line("sweep", removal, "nothing can read it");
    }
    for removal in &plan.tiers {
        line("tier", removal, "rebuilds from the source file");
    }
    for removal in &plan.entries {
        line("entry", removal, "least recently used first");
    }
    for removal in &plan.open {
        line("keep", removal, "in use, never evicted");
    }
}

fn print_outcome(outcome: &Eviction, plan: &EvictionPlan) {
    println!(
        "\nfreed    {} in {}, {} -> {}",
        human_bytes(outcome.freed),
        plural(outcome.removals(), "removal", "removals"),
        human_bytes(plan.before),
        human_bytes(outcome.after)
    );
    if !outcome.skipped.is_empty() {
        println!(
            "kept     {} that became busy after the plan",
            plural(outcome.skipped.len(), "entry", "entries")
        );
    }
    if outcome.after > plan.cap {
        println!(
            "note     {} is still over the cap: the rest is in use",
            human_bytes(outcome.after)
        );
    }
}

fn line(kind: &str, removal: &Removal, why: &str) {
    println!(
        "  {kind:<7} {:<32} {:>11}  {:<28} {}",
        removal.name,
        human_bytes(removal.bytes),
        ago(removal.last_used),
        why
    );
}

/// When an entry was last read, in words a user can act on. The cache records seconds, so
/// this is coarse on purpose.
fn ago(last_used: u64) -> String {
    let now = now();
    if last_used == 0 || last_used > now {
        return "no clock".to_string();
    }
    match now - last_used {
        0..=59 => "just now".to_string(),
        60..=3_599 => format!("{}m ago", (now - last_used) / 60),
        3_600..=86_399 => format!("{}h ago", (now - last_used) / 3_600),
        elapsed => format!("{}d ago", elapsed / 86_400),
    }
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 * 100.0 / whole as f64
}

fn plural(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_no_cap_when_parsed_then_it_is_the_configured_twenty_gigabytes() {
        assert_eq!(cap_bytes(None).expect("a default cap"), DEFAULT_CAP_BYTES);
        assert_eq!(cap_bytes(None).expect("a default cap"), 20_000_000_000);
    }

    #[test]
    fn given_a_cap_in_gigabytes_when_parsed_then_it_is_that_many_bytes() {
        assert_eq!(cap_bytes(Some(1.5)).expect("a cap"), 1_500_000_000);
        assert_eq!(cap_bytes(Some(0.0)).expect("an empty cache"), 0);
    }

    #[test]
    fn given_a_cap_that_is_not_a_size_when_parsed_then_it_is_an_error() {
        for cap in [-1.0, f64::NAN, f64::INFINITY] {
            let error = cap_bytes(Some(cap)).expect_err("not a size");
            assert!(format!("{error:#}").contains("--cap-gb"), "{error:#}");
        }
    }

    #[test]
    fn given_a_time_when_described_then_it_reads_as_an_age_and_zero_has_no_clock() {
        assert_eq!(ago(0), "no clock");
        assert_eq!(ago(now() - 5), "just now");
        assert_eq!(ago(now() - 120), "2m ago");
        assert_eq!(ago(now() - 7_200), "2h ago");
        assert_eq!(ago(now() - 3 * 86_400), "3d ago");
        assert_eq!(
            ago(now() + 60),
            "no clock",
            "a clock in the future is no clock"
        );
    }
}
