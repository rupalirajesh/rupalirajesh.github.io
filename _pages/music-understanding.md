---
layout: single
title: "Where Music Understanding Breaks in Audio-Language Models"
permalink: /research/music-understanding/
author_profile: true
toc: true
toc_sticky: true
---

*Ongoing research, 2026. Code and full data: [github.com/rupalirajesh/music-understanding](https://github.com/rupalirajesh/music-understanding).*

## The question

Audio-language models (LALMs) — Qwen2-Audio, Gemini's native audio mode, Audio Flamingo,
and similar — take in audio and text and answer questions about what they hear. They're
increasingly good at describing music: genre, mood, instrumentation. But description isn't
the same as *perception*. A well-known result (MuChoMusic, ISMIR 2024) caught models scoring
respectably on music benchmarks *even with the audio removed* — meaning the questions were
answerable from text plausibility alone, not from listening.

This project asks, task by task: **can these models actually hear pitch, key, tempo, and
meter — or are they guessing from priors?** And when they fail, *where* in the model does
the failure live — is the information never captured by the audio encoder, captured but not
passed through to language, or captured and passable but the model just wasn't trained to
say it?

## Method, briefly

- **Six models**, tested identically: Qwen2-Audio-7B, Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B,
  Audio Flamingo 3, Music Flamingo, and Gemini-2.5-Pro. (GPT-4o-audio pending.)
- **A synthetic stimulus battery** (MIDI + fluidsynth, perfect ground truth) across pitch,
  octave, interval, key, mode, chord, tempo, meter, and instrument-identification tasks —
  both multiple-choice and open-ended framings of the same questions.
- **Three measurement levels**, on the same stimuli, to localize *why* a task fails:
  1. **Signal** — is the property recoverable at all with classical signal processing?
  2. **Representation** — train a tiny linear classifier on the model's own audio-encoder
     vectors. If it can read off the answer, the information is physically encoded there,
     whether or not the model ever says it in words.
  3. **Behavior** — just ask the model, in plain language, and score the answer.
- **Two validity controls on every task**: a *no-audio* twin question (if accuracy barely
  drops without audio, the task was measuring text priors, not listening) and a
  *wrong-audio* twin (swap in an unrelated clip — if accuracy doesn't drop, the model isn't
  using audio content at all).

## Listen for yourself

Four examples below, one per finding. Each plays the actual test clip and shows what all
six models answered.

<div class="audio-example">
  <h4>1. The one clean, unambiguous win: instrument identification</h4>
  <audio controls preload="none" src="/assets/audio/instrument_piano.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>piano</strong></p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>piano</td><td class="ok">✓ correct</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>piano</td><td class="ok">✓ correct</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>piano</td><td class="ok">✓ correct</td></tr>
      <tr><td>Audio Flamingo 3</td><td>piano</td><td class="ok">✓ correct</td></tr>
      <tr><td>Music Flamingo</td><td>piano</td><td class="ok">✓ correct</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>piano</td><td class="ok">✓ correct</td></tr>
    </tbody>
  </table>
  <p class="audio-example__note">Every model gets this right, and it isn't a fluke: every
  model shows a large accuracy <em>gain</em> from having the audio at all (+16 to +71 points
  over the no-audio control), and a linear probe recovers the instrument from the raw audio
  encoder at 91–94% accuracy. Signal, representation, and behavior all agree — use this task
  as the sanity-check reference when judging the others.</p>
</div>

<div class="audio-example">
  <h4>2. A real alignment gap: octave placement</h4>
  <audio controls preload="none" src="/assets/audio/octave_flute.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>octave 3</strong></p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>3</td><td class="ok">✓ correct</td></tr>
      <tr><td>Audio Flamingo 3</td><td>2</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Music Flamingo</td><td>3</td><td class="ok">✓ correct</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>3</td><td class="ok">✓ correct</td></tr>
    </tbody>
  </table>
  <p class="audio-example__note">Half the models miss by exactly one octave — the classic
  "right note, wrong register" error. Across the full battery, octave placement is one of
  the clearest <strong>alignment-gap</strong> cases: a linear probe reads it off the raw
  encoder at 99% accuracy, but average behavioral accuracy sits around 40%. The ear has it;
  the model doesn't reliably say it.</p>
</div>

<div class="audio-example">
  <h4>3. A task that looks broken, not just hard: beats per bar</h4>
  <audio controls preload="none" src="/assets/audio/beats_waltz.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>3 beats per bar</strong> (a waltz)</p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Audio Flamingo 3</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Music Flamingo</td><td>4</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>4</td><td class="bad">✗ wrong</td></tr>
    </tbody>
  </table>
  <p class="audio-example__note">All six models say "4." That's not a coincidence — it shows
  up even on the <em>no-audio</em> control question (same guess, no clip at all), and
  swapping in a completely unrelated clip doesn't change the answer either: average accuracy
  is actually <em>higher</em> with the wrong audio than the correct audio, the only task in
  the battery where that happens. This looks like "4/4 time is the most common answer,"
  not a hearing failure — a task-validity flag, not a capability finding.</p>
</div>

<div class="audio-example">
  <h4>4. When multiple-choice inflates the score: key identification</h4>
  <audio controls preload="none" src="/assets/audio/key_cmajor.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>C major</strong></p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>F major</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>C major</td><td class="ok">✓ correct</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>A minor</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Audio Flamingo 3</td><td>A minor</td><td class="bad">✗ wrong</td></tr>
      <tr><td>Music Flamingo</td><td>C major</td><td class="ok">✓ correct</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>C major</td><td class="ok">✓ correct</td></tr>
    </tbody>
  </table>
  <p class="audio-example__note">The wrong answers here aren't random — F major and A minor
  are both one step away from C major (a fifth, and the relative minor). That looks like
  evidence of real listening, <em>except</em> the multiple-choice options for this task are
  deliberately built from exactly those "musically close" alternatives, so a structured-looking
  wrong answer is partly built into the question, not just discovered by the model. Rerunning
  the comparison on open-ended (non-multiple-choice) versions of this question is more telling:
  Gemini's key-identification accuracy — a respectable ~42% on multiple choice — drops to
  <strong>0 out of 20</strong> once it has to name the key itself with no options to choose
  from. Its multiple-choice score was mostly a scaffolding effect, not a key-naming skill.</p>
</div>

## The broader pattern

Putting the behavioral battery, the linear probes, and the controls together:

- **No-audio "refusal" behavior is not comparable across models as a single number.**
  Gemini-2.5-Pro explicitly refuses 57% of no-audio questions ("I can't hear anything").
  Qwen2-Audio refuses 13%. The other four models refuse **0%** — but that's not because
  they're more capable; spot-checking their answers shows they guess a specific, confident
  answer regardless. Zero refusals reads as "more helpful" at a glance and is actually the
  more concerning failure mode.
- **The wrong-audio control is the cleanest single signal for "is this model actually
  listening."** Music Flamingo shows the largest accuracy drop when the correct clip is
  swapped for an unrelated one (+10.8 points better with correct audio) — the strongest
  evidence of genuine listening among the six. Qwen2.5-Omni shows *no* drop at all
  (−1.0 points) — its answers don't depend on which clip it's given.
- **Encoder architecture predicts what survives, independent of what the encoder was
  trained to do.** MERT (trained on music specifically) and the Whisper encoder (trained
  for speech transcription) agree on very little in objective — but both are *frame-level*
  encoders, producing one vector every ~15–20ms, and both preserve fine pitch/timing detail
  well. CLAP, a *clip-level* encoder built for caption-matching, pools the whole clip into
  one global vector and is markedly worse at everything time-local. The architecture of
  *how* an encoder compresses audio matters as much as what it was trained on.
- **GPT-4o-audio is the one model still pending** in the current battery; the harness and
  every control described here apply to it unchanged once it's run.

## What I'm doing next

- Re-probing each model's *own* audio encoder (not a generic standalone one) on the tasks
  where behavior currently beats the probe — the biggest open question in the current data.
- A representation "ladder": the same questions given progressively more explicit
  information (raw audio → audio + extracted pitch/beat features → symbolic notation) to
  pin down whether a failure is architectural or just a training-data gap.
- Small LoRA fine-tunes on the clearest "the information is there, the model just doesn't
  say it" cases, to test whether that gap is cheaply closeable.

Full methodology, all raw data, and the complete capability tables are in the
[project repository](https://github.com/rupalirajesh/music-understanding), including
`RESEARCH_PLAN.md` (the full study design) and `PAPER.md` (results as they land).
