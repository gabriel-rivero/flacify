//! The blind A/B run: a seeded plan, per-trial answers, a level gate and the report.
//!
//! The same seed asks the same excerpts in the same order, so a report can be replayed and
//! checked. The gate matters as much as the score: two sides at different loudness measure
//! the loudness, not the model.

use anyhow::{bail, ensure, Context, Result};
use flacify_core::SAMPLE_RATE;

use crate::loudness;

/// Length of each excerpt, in seconds.
pub(crate) const EXCERPT_SECS: f64 = 6.0;

/// The most the two sides of one excerpt may differ in integrated loudness.
pub(crate) const MAX_LEVEL_DIFF_DB: f64 = 0.5;

/// The 95 % normal quantile, for the Wilson interval.
const Z_95: f64 = 1.959_963_984_540_054;

/// Which of a trial's two excerpts holds the generated side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Position {
    First,
    Second,
}

/// One of the two signals.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Side {
    Original,
    Generated,
}

/// The order one trial plays its sides in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Order {
    OriginalFirst,
    GeneratedFirst,
}

impl Order {
    /// The side the operator hears first.
    pub(crate) fn first(self) -> Side {
        match self {
            Self::OriginalFirst => Side::Original,
            Self::GeneratedFirst => Side::Generated,
        }
    }

    /// The side the operator hears second.
    pub(crate) fn second(self) -> Side {
        match self.first() {
            Side::Original => Side::Generated,
            Side::Generated => Side::Original,
        }
    }

    /// Where the generated side plays, which is the answer key.
    fn truth(self) -> Position {
        match self {
            Self::OriginalFirst => Position::Second,
            Self::GeneratedFirst => Position::First,
        }
    }
}

/// One excerpt pair: where it starts, its order, the answer and the answer key.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Trial {
    pub(crate) index: u32,
    /// Start of the excerpt, in frames at 48 kHz.
    pub(crate) excerpt_start: usize,
    pub(crate) order: Order,
    /// Where the generated side played; `answer` is compared with it.
    pub(crate) truth: Position,
    pub(crate) answer: Option<Position>,
    pub(crate) correct: Option<bool>,
}

impl Trial {
    fn new(index: u32, excerpt_start: usize, order: Order) -> Self {
        Self {
            index,
            excerpt_start,
            order,
            truth: order.truth(),
            answer: None,
            correct: None,
        }
    }
}

/// The trials one seed asks, in the order it asks them.
#[derive(Debug)]
pub(crate) struct Plan {
    seed: u64,
    excerpt_frames: usize,
    trials: Vec<Trial>,
}

impl Plan {
    /// Lay out `trial_count` non-overlapping excerpts and draw each one's order.
    ///
    /// Each trial gets its own slice of the track, so no content is heard twice, and the
    /// start within the slice is drawn from the seed. Fails when the track cannot hold the
    /// trials at this excerpt length: the caller passes a smaller `--trials`.
    pub(crate) fn new(
        seed: u64,
        trial_count: u32,
        track_frames: usize,
        excerpt_frames: usize,
    ) -> Result<Self> {
        ensure!(trial_count > 0, "a blind run needs at least one trial");
        ensure!(
            excerpt_frames > 0,
            "an excerpt of zero frames cannot be played"
        );
        let count = trial_count as usize;
        let needed = count.saturating_mul(excerpt_frames);
        ensure!(
            track_frames >= needed,
            "{trial_count} trials of {excerpt_frames} frames need {needed} frames; \
             the track has {track_frames}, so pass a smaller --trials"
        );

        let slot = track_frames / count;
        let slack = slot - excerpt_frames;
        let mut rng = SplitMix64::new(seed);
        let mut trials = Vec::with_capacity(count);
        for (index, order) in draw_orders(count, &mut rng).into_iter().enumerate() {
            let jitter = match slack {
                0 => 0,
                _ => (rng.next_u64() % (slack as u64 + 1)) as usize,
            };
            trials.push(Trial::new(index as u32, index * slot + jitter, order));
        }
        Ok(Self {
            seed,
            excerpt_frames,
            trials,
        })
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) fn excerpt_frames(&self) -> usize {
        self.excerpt_frames
    }

    pub(crate) fn trials(&self) -> &[Trial] {
        &self.trials
    }

    /// Record what the operator said for one trial, and score it against the truth.
    pub(crate) fn record(&mut self, index: u32, answer: Position) -> Result<()> {
        let total = self.trials.len();
        let trial = self
            .trials
            .get_mut(index as usize)
            .with_context(|| format!("trial {} is not in a plan of {total} trials", index + 1))?;
        trial.answer = Some(answer);
        trial.correct = Some(answer == trial.truth);
        Ok(())
    }

    /// The answered trials, or `None` when the operator answered none.
    pub(crate) fn score(&self) -> Option<Score> {
        let answered = self
            .trials
            .iter()
            .filter(|trial| trial.correct.is_some())
            .count() as u32;
        if answered == 0 {
            return None;
        }
        let correct = self
            .trials
            .iter()
            .filter(|trial| trial.correct == Some(true))
            .count() as u32;
        let proportion = f64::from(correct) / f64::from(answered);
        let (ci_low, ci_high) = wilson_interval(proportion, answered);
        Some(Score {
            correct,
            answered,
            p_value: binomial_tail(answered, correct),
            ci_low,
            ci_high,
        })
    }
}

/// What the answers add up to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Score {
    pub(crate) correct: u32,
    pub(crate) answered: u32,
    /// One-sided probability of at least this many right answers from guessing.
    pub(crate) p_value: f64,
    /// Wilson 95 % interval for the hit rate, as proportions.
    pub(crate) ci_low: f64,
    pub(crate) ci_high: f64,
}

/// The two sides of one excerpt, measured as BS.1770 integrated loudness.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelCheck {
    pub(crate) index: u32,
    pub(crate) excerpt_start: usize,
    pub(crate) original_lufs: f64,
    pub(crate) generated_lufs: f64,
    /// `generated - original`: positive when the generated side is louder.
    pub(crate) delta_db: f64,
}

impl LevelCheck {
    /// False when a side is silent, so the difference is not a number.
    pub(crate) fn is_measurable(&self) -> bool {
        self.delta_db.is_finite()
    }

    pub(crate) fn is_matched(&self) -> bool {
        self.is_measurable() && self.delta_db.abs() <= MAX_LEVEL_DIFF_DB
    }
}

/// Measure one excerpt on both sides. A silent side gives `-inf` and a `delta_db` that is
/// not finite, which the gate treats as a failed measurement rather than a match.
pub(crate) fn measure_level(
    trial: &Trial,
    original: &[&[f32]],
    generated: &[&[f32]],
) -> LevelCheck {
    let original_lufs = loudness::integrated(original);
    let generated_lufs = loudness::integrated(generated);
    LevelCheck {
        index: trial.index,
        excerpt_start: trial.excerpt_start,
        original_lufs,
        generated_lufs,
        delta_db: generated_lufs - original_lufs,
    }
}

/// Whether every excerpt is inside the gate. An empty check list is not a match.
pub(crate) fn levels_are_matched(levels: &[LevelCheck]) -> bool {
    !levels.is_empty() && levels.iter().all(LevelCheck::is_matched)
}

/// The 0.5 dB gate the run must pass before its score means anything.
pub(crate) fn level_gate(levels: &[LevelCheck]) -> Result<()> {
    if levels.is_empty() {
        bail!("no excerpt was level-checked, so the sides cannot be compared");
    }
    for check in levels {
        let seconds = check.excerpt_start as f64 / f64::from(SAMPLE_RATE);
        if !check.is_measurable() {
            bail!(
                "trial {} at {seconds:.3} s has a silent side ({:.1} / {:.1} LUFS), \
                 so its level cannot be checked",
                check.index + 1,
                check.original_lufs,
                check.generated_lufs
            );
        }
        if !check.is_matched() {
            bail!(
                "trial {} at {seconds:.3} s is out of level by {:+.2} dB, over the \
                 {MAX_LEVEL_DIFF_DB} dB gate; the score would measure loudness",
                check.index + 1,
                check.delta_db
            );
        }
    }
    Ok(())
}

/// The largest difference in the list, for the line the report quotes.
pub(crate) fn worst_level_diff(levels: &[LevelCheck]) -> Option<f64> {
    levels
        .iter()
        .filter(|check| check.is_measurable())
        .map(|check| check.delta_db.abs())
        .reduce(f64::max)
}

/// What one report says about the run.
pub(crate) struct ReportMeta<'a> {
    pub(crate) file: &'a str,
    pub(crate) device: &'a str,
    pub(crate) model: &'a str,
    pub(crate) sample_rate: u32,
}

/// The Markdown report: the seed to replay, the measured level match, every trial with its
/// answer and the truth, and the score.
pub(crate) fn markdown(meta: &ReportMeta<'_>, plan: &Plan, levels: &[LevelCheck]) -> String {
    let mut text = String::new();
    text.push_str("# Blind A/B report\n\n");
    text.push_str("| field | value |\n|---|---|\n");
    text.push_str(&format!("| file | `{}` |\n", meta.file));
    text.push_str(&format!("| device | `{}` |\n", meta.device));
    text.push_str(&format!("| model | `{}` |\n", meta.model));
    text.push_str(&format!("| seed | `{}` |\n", plan.seed()));
    text.push_str(&format!(
        "| excerpt | {:.3} s |\n",
        plan.excerpt_frames() as f64 / f64::from(meta.sample_rate)
    ));
    text.push_str(&format!("| trials | {} |\n", plan.trials().len()));
    text.push_str("\nThe same seed asks the same excerpts in the same order:\n\n");
    text.push_str(&format!(
        "```sh\nflacify ab {} --trials {} --seed {}\n```\n\n",
        meta.file,
        plan.trials().len(),
        plan.seed()
    ));

    text.push_str("## Level match\n\n");
    text.push_str(
        "Each excerpt measured on both sides as BS.1770 integrated loudness. A run is \
         evidence only if every pair is within 0.500 dB.\n\n",
    );
    text.push_str(
        "| trial | excerpt | original (LUFS) | generated (LUFS) | difference |\n\
         |---:|---|---:|---:|---:|\n",
    );
    for check in levels {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            check.index + 1,
            time(check.excerpt_start, meta.sample_rate),
            lufs(check.original_lufs),
            lufs(check.generated_lufs),
            difference(check.delta_db),
        ));
    }
    text.push('\n');
    text.push_str(&level_verdict(levels, meta.sample_rate));
    text.push_str("\n\n");

    text.push_str("## Trials\n\n");
    let score = plan.score();
    match score {
        None => {
            let reason = match levels_are_matched(levels) {
                true => "No trial was answered.",
                false => "The trials did not run: the level gate above failed.",
            };
            text.push_str(reason);
            text.push_str("\n\n");
        }
        Some(_) => {
            text.push_str(
                "| trial | excerpt | generated | answer | correct |\n\
                 |---:|---|---|---|---|\n",
            );
            for trial in plan.trials() {
                text.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    trial.index + 1,
                    time(trial.excerpt_start, meta.sample_rate),
                    position(trial.truth),
                    trial.answer.map_or_else(|| "-".to_string(), position),
                    match trial.correct {
                        Some(true) => "yes",
                        Some(false) => "no",
                        None => "-",
                    },
                ));
            }
            text.push('\n');
            text.push_str(
                "`generated` is where the generated side played, which is the truth; \
                 `answer` is what the operator said.\n\n",
            );
        }
    }

    text.push_str("## Score\n\n");
    match score {
        Some(score) => {
            text.push_str(&format!(
                "{} of {} correct ({:.1}%).\n",
                score.correct,
                score.answered,
                100.0 * f64::from(score.correct) / f64::from(score.answered)
            ));
            text.push_str(&format!(
                "One-sided binomial p-value at chance (p = 0.5): {}.\n",
                p_label(score.p_value)
            ));
            text.push_str(&format!(
                "95% confidence interval (Wilson): {:.1}% to {:.1}%.\n",
                score.ci_low * 100.0,
                score.ci_high * 100.0
            ));
        }
        None => text.push_str("No trial was scored.\n"),
    }
    text.push_str(
        "\nA trial plays the same excerpt twice, once per side, in the order in the table. \
         The score is a statement about this listener on this pair of files, nothing wider.\n",
    );
    text
}

/// What one line of operator input means.
pub(crate) fn parse_answer(line: &str) -> Option<Position> {
    match line.trim().to_ascii_lowercase().as_str() {
        "1" | "first" => Some(Position::First),
        "2" | "second" => Some(Position::Second),
        _ => None,
    }
}

/// The p-value as a reader can compare it: four decimals until that loses the magnitude.
pub(crate) fn p_label(p_value: f64) -> String {
    match p_value >= 1e-4 {
        true => format!("{p_value:.4}"),
        false => format!("{p_value:.2e}"),
    }
}

fn level_verdict(levels: &[LevelCheck], sample_rate: u32) -> String {
    if levels.is_empty() {
        return "**Not measured**: no excerpt was checked, so the run is not evidence.".to_string();
    }
    if let Some(check) = levels.iter().find(|check| !check.is_measurable()) {
        return format!(
            "**Not measured**: trial {} at {} has a silent side, so its level cannot be \
             checked. This run is not evidence.",
            check.index + 1,
            time(check.excerpt_start, sample_rate)
        );
    }
    if let Some(check) = levels.iter().find(|check| !check.is_matched()) {
        return format!(
            "**Not matched**: trial {} at {} differs by {:+.2} dB, over the \
             {MAX_LEVEL_DIFF_DB:.3} dB gate. This run is not evidence.",
            check.index + 1,
            time(check.excerpt_start, sample_rate),
            check.delta_db
        );
    }
    let mut worst = &levels[0];
    for check in levels {
        if check.delta_db.abs() > worst.delta_db.abs() {
            worst = check;
        }
    }
    format!(
        "Worst difference {:+.2} dB at trial {}: all {} excerpts are within the \
         {MAX_LEVEL_DIFF_DB:.3} dB gate.",
        worst.delta_db,
        worst.index + 1,
        levels.len()
    )
}

fn time(frames: usize, sample_rate: u32) -> String {
    format!("{:.3} s", frames as f64 / f64::from(sample_rate))
}

fn lufs(value: f64) -> String {
    match value.is_finite() {
        true => format!("{value:.2}"),
        false => "-inf".to_string(),
    }
}

fn difference(delta_db: f64) -> String {
    match delta_db.is_finite() {
        true => format!("{delta_db:+.2} dB"),
        false => "not measurable".to_string(),
    }
}

fn position(position: Position) -> String {
    match position {
        Position::First => "first".to_string(),
        Position::Second => "second".to_string(),
    }
}

/// P(X >= `successes`) for a fair coin flipped `trials` times, in log space so the middle
/// of a twenty-trial run does not need 2^20 as a float.
fn binomial_tail(trials: u32, successes: u32) -> f64 {
    if successes == 0 {
        return 1.0;
    }
    if successes > trials {
        return 0.0;
    }
    let mut log_term = log_choose(trials, successes) - f64::from(trials) * std::f64::consts::LN_2;
    let mut total = 0.0;
    for index in successes..=trials {
        total += log_term.exp();
        if index < trials {
            log_term += (f64::from(trials - index) / (f64::from(index) + 1.0)).ln();
        }
    }
    total.min(1.0)
}

fn log_choose(n: u32, k: u32) -> f64 {
    let k = k.min(n - k);
    (0..k)
        .map(|index| (f64::from(n - index)).ln() - (f64::from(index) + 1.0).ln())
        .sum()
}

fn wilson_interval(proportion: f64, trials: u32) -> (f64, f64) {
    let n = f64::from(trials);
    let z_squared = Z_95 * Z_95;
    let denominator = 1.0 + z_squared / n;
    let centre = proportion + z_squared / (2.0 * n);
    let margin = Z_95 * (proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n)).sqrt();
    (
        (centre - margin) / denominator,
        (centre + margin) / denominator,
    )
}

/// `trial_count / 2` trials hear the generated side first, the rest hear it second, and the
/// seeded shuffle decides which trial gets which. Exact balance, still random.
fn draw_orders(count: usize, rng: &mut SplitMix64) -> Vec<Order> {
    let mut orders = vec![Order::GeneratedFirst; count / 2];
    orders.resize(count, Order::OriginalFirst);
    for index in (1..count).rev() {
        let swap = (rng.next_u64() % (index as u64 + 1)) as usize;
        orders.swap(index, swap);
    }
    orders
}

/// SplitMix64: the seeded stream behind the excerpt starts and the orders. Small enough to
/// keep in this file, and deterministic on every platform.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten minutes at 48 kHz, the length a real track has.
    const TRACK: usize = 600 * 48_000;
    /// One excerpt: six seconds.
    const EXCERPT: usize = 6 * 48_000;

    fn opposite(position: Position) -> Position {
        match position {
            Position::First => Position::Second,
            Position::Second => Position::First,
        }
    }

    /// A level check with a given difference, without measuring any audio.
    fn check(index: u32, delta_db: f64) -> LevelCheck {
        LevelCheck {
            index,
            excerpt_start: index as usize * EXCERPT,
            original_lufs: -14.0,
            generated_lufs: -14.0 + delta_db,
            delta_db,
        }
    }

    /// A one-second 1 kHz sine at 48 kHz, on one channel.
    fn sine(amplitude: f32) -> Vec<f32> {
        (0..48_000)
            .map(|index| {
                amplitude * (std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0).sin()
            })
            .collect()
    }

    fn meta() -> ReportMeta<'static> {
        ReportMeta {
            file: "track.flac",
            device: "test device",
            model: "cpu",
            sample_rate: 48_000,
        }
    }

    #[test]
    fn given_the_same_seed_when_a_plan_is_built_twice_then_every_trial_matches() {
        let first = Plan::new(7, 20, TRACK, EXCERPT).expect("a plan");
        let second = Plan::new(7, 20, TRACK, EXCERPT).expect("a plan");
        assert_eq!(first.trials(), second.trials());
    }

    #[test]
    fn given_two_seeds_when_plans_are_built_then_the_excerpts_and_the_orders_differ() {
        let first = Plan::new(1, 20, TRACK, EXCERPT).expect("a plan");
        let second = Plan::new(2, 20, TRACK, EXCERPT).expect("a plan");
        assert_ne!(first.trials(), second.trials());
        let starts = |plan: &Plan| -> Vec<usize> {
            plan.trials()
                .iter()
                .map(|trial| trial.excerpt_start)
                .collect()
        };
        assert_ne!(starts(&first), starts(&second));
        let orders =
            |plan: &Plan| -> Vec<Order> { plan.trials().iter().map(|trial| trial.order).collect() };
        assert_ne!(
            orders(&first),
            orders(&second),
            "the order comes from the seed, not from the trial number"
        );
    }

    #[test]
    fn given_a_plan_when_examined_then_the_excerpts_are_inside_the_track_and_never_overlap() {
        let plan = Plan::new(11, 20, TRACK, EXCERPT).expect("a plan");
        let mut previous_end = 0;
        for trial in plan.trials() {
            assert!(
                trial.excerpt_start >= previous_end,
                "trial {} overlaps the one before it",
                trial.index + 1
            );
            assert!(trial.excerpt_start + EXCERPT <= TRACK);
            previous_end = trial.excerpt_start + EXCERPT;
        }
    }

    #[test]
    fn given_a_track_exactly_fitting_the_trials_when_built_then_the_starts_touch_back_to_back() {
        let plan = Plan::new(3, 20, 20 * EXCERPT, EXCERPT).expect("a plan");
        for (index, trial) in plan.trials().iter().enumerate() {
            assert_eq!(trial.excerpt_start, index * EXCERPT);
        }
    }

    #[test]
    fn given_a_plan_when_the_orders_are_counted_then_the_two_orders_are_balanced_and_both_occur() {
        let plan = Plan::new(3, 20, TRACK, EXCERPT).expect("a plan");
        let generated_first = plan
            .trials()
            .iter()
            .filter(|trial| trial.order == Order::GeneratedFirst)
            .count();
        assert_eq!(generated_first, 10);

        // An odd trial count rounds the generated-first half down, never up.
        let odd = Plan::new(3, 21, TRACK, EXCERPT).expect("a plan");
        assert_eq!(
            odd.trials()
                .iter()
                .filter(|trial| trial.order == Order::GeneratedFirst)
                .count(),
            10
        );
    }

    #[test]
    fn given_an_order_when_asked_then_each_position_is_a_different_side() {
        assert_eq!(Order::GeneratedFirst.first(), Side::Generated);
        assert_eq!(Order::GeneratedFirst.second(), Side::Original);
        assert_eq!(Order::OriginalFirst.first(), Side::Original);
        assert_eq!(Order::OriginalFirst.second(), Side::Generated);
    }

    #[test]
    fn given_a_plan_when_the_truth_is_read_then_it_is_where_the_generated_side_plays() {
        let plan = Plan::new(4, 20, TRACK, EXCERPT).expect("a plan");
        for trial in plan.trials() {
            let expected = match trial.order {
                Order::GeneratedFirst => Position::First,
                Order::OriginalFirst => Position::Second,
            };
            assert_eq!(trial.truth, expected);
        }
    }

    #[test]
    fn given_a_track_too_short_for_the_trials_when_built_then_the_error_names_the_shortfall() {
        let error = Plan::new(1, 20, 100 * 48_000, EXCERPT)
            .expect_err("100 s cannot hold 20 excerpts")
            .to_string();
        assert!(error.contains("need 5760000 frames"), "{error}");
        assert!(error.contains("has 4800000"), "{error}");
        assert!(error.contains("--trials"), "{error}");
    }

    #[test]
    fn given_zero_trials_when_built_then_it_is_rejected() {
        let error = Plan::new(1, 0, TRACK, EXCERPT)
            .expect_err("a plan needs a trial")
            .to_string();
        assert!(error.contains("at least one trial"), "{error}");
    }

    #[test]
    fn given_a_zero_length_excerpt_when_built_then_it_is_rejected() {
        let error = Plan::new(1, 20, TRACK, 0)
            .expect_err("an empty excerpt cannot be played")
            .to_string();
        assert!(error.contains("zero frames"), "{error}");
    }

    #[test]
    fn given_a_trial_outside_the_plan_when_recorded_then_it_is_an_error() {
        let mut plan = Plan::new(1, 4, 40 * 48_000, EXCERPT).expect("a plan");
        let error = plan
            .record(4, Position::First)
            .expect_err("trial 5 is not in a plan of four")
            .to_string();
        assert!(error.contains("trial 5"), "{error}");
        assert!(error.contains("4 trials"), "{error}");
    }

    #[test]
    fn given_scripted_answers_when_scored_then_the_count_p_value_and_interval_are_the_binomial_ones(
    ) {
        let mut plan = Plan::new(5, 20, TRACK, EXCERPT).expect("a plan");
        let truths: Vec<Position> = plan.trials().iter().map(|trial| trial.truth).collect();
        for (index, truth) in truths.iter().enumerate() {
            let answer = match index < 15 {
                true => *truth,
                false => opposite(*truth),
            };
            plan.record(index as u32, answer)
                .expect("a recorded answer");
        }

        let score = plan.score().expect("a score");
        assert_eq!(score.correct, 15);
        assert_eq!(score.answered, 20);
        assert!(
            (score.p_value - 0.020_695).abs() < 1e-5,
            "p = {}",
            score.p_value
        );
        assert!((score.ci_low - 0.5313).abs() < 1e-3, "{}", score.ci_low);
        assert!((score.ci_high - 0.8881).abs() < 1e-3, "{}", score.ci_high);
    }

    #[test]
    fn given_no_answers_when_scored_then_there_is_no_score() {
        let plan = Plan::new(5, 20, TRACK, EXCERPT).expect("a plan");
        assert!(plan.score().is_none());
    }

    #[test]
    fn given_a_partial_run_when_scored_then_unanswered_trials_are_not_in_the_denominator() {
        let mut plan = Plan::new(5, 20, TRACK, EXCERPT).expect("a plan");
        let truths: Vec<Position> = plan.trials().iter().map(|trial| trial.truth).collect();
        for (index, truth) in truths.iter().enumerate().take(4) {
            plan.record(index as u32, *truth)
                .expect("a recorded answer");
        }
        let score = plan.score().expect("a score");
        assert_eq!(score.correct, 4);
        assert_eq!(score.answered, 4);
    }

    #[test]
    fn given_every_trial_right_when_scored_then_the_interval_ends_at_one() {
        let mut plan = Plan::new(6, 20, TRACK, EXCERPT).expect("a plan");
        let truths: Vec<Position> = plan.trials().iter().map(|trial| trial.truth).collect();
        for (index, truth) in truths.iter().enumerate() {
            plan.record(index as u32, *truth)
                .expect("a recorded answer");
        }
        let score = plan.score().expect("a score");
        assert_eq!(score.correct, 20);
        assert!((score.ci_low - 0.8389).abs() < 1e-3, "{}", score.ci_low);
        assert!(
            (score.ci_high - 1.0).abs() < 1e-9,
            "upper {} must not pass 1",
            score.ci_high
        );
    }

    #[test]
    fn given_no_successes_when_the_tail_is_taken_then_it_is_certain() {
        // The log-space sum lands a bit under 1.0 here, so the shortcut is the answer.
        assert_eq!(binomial_tail(20, 0), 1.0);
        assert_eq!(binomial_tail(1_000, 0), 1.0);
    }

    #[test]
    fn given_more_successes_than_trials_when_the_tail_is_taken_then_it_is_impossible() {
        assert_eq!(binomial_tail(20, 21), 0.0);
    }

    #[test]
    fn given_every_trial_correct_when_the_tail_is_taken_then_it_is_one_in_two_to_the_twenty() {
        assert!((binomial_tail(20, 20) - 2f64.powi(-20)).abs() < 1e-12);
    }

    #[test]
    fn given_a_thousand_trials_when_the_tail_is_taken_then_it_stays_a_probability() {
        let p_value = binomial_tail(1_000, 600);
        assert!(p_value.is_finite());
        assert!((0.0..=1.0).contains(&p_value), "{p_value}");
        assert!(p_value < 1e-9, "{p_value}");
    }

    #[test]
    fn given_a_generated_side_four_tenths_louder_when_measured_then_the_gate_passes() {
        let original = sine(0.5);
        let gain = 10f64.powf(0.4 / 20.0) as f32;
        let generated: Vec<f32> = original.iter().map(|sample| sample * gain).collect();
        let trial = Trial::new(0, 0, Order::OriginalFirst);

        let check = measure_level(&trial, &[&original], &[&generated]);
        assert!(
            (check.delta_db - 0.4).abs() < 0.01,
            "delta {}",
            check.delta_db
        );
        assert!(check.is_matched());
        assert!(level_gate(&[check]).is_ok());
    }

    #[test]
    fn given_a_generated_side_six_tenths_louder_when_gated_then_the_run_fails_and_names_it() {
        let original = sine(0.5);
        let gain = 10f64.powf(0.6 / 20.0) as f32;
        let generated: Vec<f32> = original.iter().map(|sample| sample * gain).collect();
        let trial = Trial::new(3, 18 * 48_000, Order::GeneratedFirst);

        let check = measure_level(&trial, &[&original], &[&generated]);
        let error = level_gate(&[check]).expect_err("0.6 dB is over the gate");
        let error = error.to_string();
        assert!(error.contains("trial 4"), "{error}");
        assert!(error.contains("+0.60 dB"), "{error}");
        assert!(error.contains("0.5 dB gate"), "{error}");
    }

    #[test]
    fn given_exactly_the_gate_when_checked_then_it_passes_and_anything_over_it_fails() {
        assert!(check(0, MAX_LEVEL_DIFF_DB).is_matched());
        assert!(check(1, -MAX_LEVEL_DIFF_DB).is_matched());
        assert!(!check(2, MAX_LEVEL_DIFF_DB + 1e-6).is_matched());
        assert!(!check(3, -MAX_LEVEL_DIFF_DB - 1e-6).is_matched());
    }

    #[test]
    fn given_a_silent_side_when_measured_then_the_check_is_not_measurable_and_the_gate_fails() {
        let silence = vec![0.0f32; 4_800];
        let trial = Trial::new(2, 0, Order::GeneratedFirst);

        let check = measure_level(&trial, &[&silence], &[&silence]);
        assert!(!check.is_measurable());
        assert!(!check.is_matched());
        let error = level_gate(&[check]).expect_err("silence has no level");
        assert!(error.to_string().contains("silent side"), "{error}");
    }

    #[test]
    fn given_no_level_checks_when_gated_then_the_run_fails() {
        let error = level_gate(&[]).expect_err("nothing measured is not a match");
        assert!(error.to_string().contains("no excerpt was level-checked"));
        assert!(!levels_are_matched(&[]));
    }

    #[test]
    fn given_level_differences_when_asked_for_the_worst_then_the_largest_absolute_one_comes_back() {
        let levels = [check(0, 0.2), check(1, -0.4), check(2, 0.1)];
        assert!((worst_level_diff(&levels).expect("a worst") - 0.4).abs() < 1e-12);
        assert!(worst_level_diff(&[]).is_none());
        assert!(
            worst_level_diff(&[check(0, f64::NAN)]).is_none(),
            "an unmeasurable pair has no worst difference to quote"
        );
    }

    #[test]
    fn given_a_scripted_run_when_rendered_then_the_report_states_the_seed_the_score_and_the_levels()
    {
        let mut plan = Plan::new(1, 4, 40 * 48_000, EXCERPT).expect("a plan");
        let truths: Vec<Position> = plan.trials().iter().map(|trial| trial.truth).collect();
        for (index, truth) in truths.iter().enumerate() {
            let answer = match index < 3 {
                true => *truth,
                false => opposite(*truth),
            };
            plan.record(index as u32, answer)
                .expect("a recorded answer");
        }
        let levels = [
            check(0, 0.02),
            check(1, -0.11),
            check(2, 0.31),
            check(3, -0.04),
        ];

        let text = markdown(&meta(), &plan, &levels);
        assert!(text.contains("| seed | `1` |"), "{text}");
        assert!(text.contains("--trials 4 --seed 1"), "{text}");
        assert!(
            text.contains("Worst difference +0.31 dB at trial 3"),
            "{text}"
        );
        assert!(text.contains("3 of 4 correct (75.0%)"), "{text}");
        assert!(text.contains("0.3125"), "{text}");
        assert!(text.contains("30.1% to 95.4%"), "{text}");
        assert!(text.contains("| yes |"), "{text}");
        assert!(text.contains("| no |"), "{text}");
        assert!(
            text.contains("`generated` is where the generated side played"),
            "{text}"
        );
    }

    #[test]
    fn given_unmatched_levels_when_rendered_then_the_report_says_the_run_is_not_evidence() {
        let plan = Plan::new(2, 4, 40 * 48_000, EXCERPT).expect("a plan");
        let levels = [check(0, 0.02), check(1, 0.61)];

        let text = markdown(&meta(), &plan, &levels);
        assert!(text.contains("**Not matched**"), "{text}");
        assert!(text.contains("+0.61 dB"), "{text}");
        assert!(text.contains("The trials did not run"), "{text}");
        assert!(text.contains("No trial was scored."), "{text}");
        assert!(!text.contains("correct ("), "{text}");
    }

    #[test]
    fn given_an_unmeasurable_pair_when_rendered_then_the_report_says_so() {
        let plan = Plan::new(2, 4, 40 * 48_000, EXCERPT).expect("a plan");
        let levels = [check(0, f64::NAN)];

        let text = markdown(&meta(), &plan, &levels);
        assert!(text.contains("**Not measured**"), "{text}");
        assert!(text.contains("silent side"), "{text}");
        assert!(text.contains("not measurable"), "{text}");
        assert!(
            markdown(&meta(), &plan, &[]).contains("**Not measured**"),
            "no measurement at all is not a match"
        );
    }

    #[test]
    fn given_small_text_when_parsed_then_only_the_documented_answers_are_accepted() {
        assert_eq!(parse_answer("1"), Some(Position::First));
        assert_eq!(parse_answer(" first \n"), Some(Position::First));
        assert_eq!(parse_answer("2"), Some(Position::Second));
        assert_eq!(parse_answer("second"), Some(Position::Second));
        assert_eq!(parse_answer("3"), None);
        assert_eq!(parse_answer(""), None);
        assert_eq!(parse_answer("maybe"), None);
    }

    #[test]
    fn given_a_tiny_p_value_when_labelled_then_it_keeps_its_magnitude() {
        assert_eq!(p_label(0.3125), "0.3125");
        assert!(p_label(2f64.powi(-20)).contains('e'));
    }

    #[test]
    fn given_the_same_seed_when_the_stream_is_drawn_twice_then_the_values_match() {
        let mut first = SplitMix64::new(42);
        let mut second = SplitMix64::new(42);
        for _ in 0..8 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        assert_ne!(SplitMix64::new(1).next_u64(), SplitMix64::new(2).next_u64());
    }
}
