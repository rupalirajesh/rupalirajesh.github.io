---
layout: single
title: "How Much Do Models Understand Music?"
date: 2026-07-22
author: Rupali Rajesh
excerpt: "Six audio-language models, tested on whether they can actually hear pitch, key, tempo, and meter - with playable audio and side-by-side answers."
---

<div class="stat-row">
  <div class="stat-tile"><span class="stat-num">6</span><span class="stat-label">models tested</span></div>
  <div class="stat-tile"><span class="stat-num">13</span><span class="stat-label">listening tasks</span></div>
  <div class="stat-tile"><span class="stat-num">4</span><span class="stat-label">examples below</span></div>
</div>

Six audio-language models - Qwen2-Audio-7B, Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B, Audio
Flamingo 3, Music Flamingo, and Gemini-2.5-Pro - tested on the same pitch, key, tempo, and
meter questions, with controls for guessing from text alone. Four results below, with the
actual audio and every model's answer.

<div class="audio-example">
  <h4>1. Instrument identification - the one clean win</h4>
  <audio controls preload="none" src="/assets/audio/instrument_piano.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>piano</strong></p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>piano</td><td class="ok">✓</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>piano</td><td class="ok">✓</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>piano</td><td class="ok">✓</td></tr>
      <tr><td>Audio Flamingo 3</td><td>piano</td><td class="ok">✓</td></tr>
      <tr><td>Music Flamingo</td><td>piano</td><td class="ok">✓</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>piano</td><td class="ok">✓</td></tr>
    </tbody>
  </table>
  <ul class="audio-example__points">
    <li>All six models correct - not a fluke.</li>
    <li>Every model gains +16 to +71 accuracy points from having the audio at all.</li>
    <li>A simple linear probe reads the instrument off the raw audio encoder at 91–94% accuracy.</li>
  </ul>
</div>

<div class="audio-example">
  <h4>2. Octave placement - a real alignment gap</h4>
  <audio controls preload="none" src="/assets/audio/octave_flute.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>octave 3</strong></p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>3</td><td class="ok">✓</td></tr>
      <tr><td>Audio Flamingo 3</td><td>2</td><td class="bad">✗</td></tr>
      <tr><td>Music Flamingo</td><td>3</td><td class="ok">✓</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>3</td><td class="ok">✓</td></tr>
    </tbody>
  </table>
  <ul class="audio-example__points">
    <li>Half the models miss by exactly one octave - right note, wrong register.</li>
    <li>The encoder probe hits 99% here. Average behavioral accuracy is ~40%.</li>
    <li>The ear has it; the model doesn't reliably say it.</li>
  </ul>
</div>

<div class="audio-example">
  <h4>3. Beats per bar - looks broken, not just hard</h4>
  <audio controls preload="none" src="/assets/audio/beats_waltz.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>3 beats per bar</strong> (a waltz)</p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Audio Flamingo 3</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Music Flamingo</td><td>4</td><td class="bad">✗</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>4</td><td class="bad">✗</td></tr>
    </tbody>
  </table>
  <ul class="audio-example__points">
    <li>All six models say "4," right or wrong.</li>
    <li>Same guess shows up with <em>no audio at all</em>.</li>
    <li>Swapping in a completely unrelated clip doesn't change the answer either - accuracy is actually higher with the wrong clip than the correct one, the only task where that happens.</li>
  </ul>
</div>

<div class="audio-example">
  <h4>4. Key identification - when multiple-choice inflates the score</h4>
  <audio controls preload="none" src="/assets/audio/key_cmajor.mp3">Your browser doesn't support inline audio.</audio>
  <p class="audio-example__caption">Ground truth: <strong>C major</strong></p>
  <table class="model-answer-table">
    <thead><tr><th>Model</th><th>Answer</th><th>Result</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-Audio-7B</td><td>F major</td><td class="bad">✗</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>C major</td><td class="ok">✓</td></tr>
      <tr><td>Qwen3-Omni-30B</td><td>A minor</td><td class="bad">✗</td></tr>
      <tr><td>Audio Flamingo 3</td><td>A minor</td><td class="bad">✗</td></tr>
      <tr><td>Music Flamingo</td><td>C major</td><td class="ok">✓</td></tr>
      <tr><td>Gemini-2.5-Pro</td><td>C major</td><td class="ok">✓</td></tr>
    </tbody>
  </table>
  <ul class="audio-example__points">
    <li>The wrong answers aren't random: F major and A minor are each one step from C major.</li>
    <li>But the multiple-choice options are built from exactly those "musically close" alternatives, so that's partly by construction, not proof of listening.</li>
    <li>Open-ended (no options given), Gemini goes from ~42% to <strong>0 out of 20</strong> on naming the key itself.</li>
  </ul>
</div>

## The broader pattern

- **"No audio present" isn't handled the same way across models.** Gemini explicitly refuses 57% of no-audio questions. Qwen2-Audio refuses 13%. The other four models refuse **0%** - but only because they guess a confident, specific answer anyway. Zero refusals looks more capable and isn't.
- **Swapping in the wrong audio clip is the cleanest "is it actually listening" test.** Music Flamingo drops the most accuracy when given the wrong clip (+10.8 points better with the correct one) - the strongest sign of real listening of the six. Qwen2.5-Omni shows no drop at all - its answers don't depend on which clip it hears.
- **What an encoder was trained to compress predicts what it keeps, more than what it was trained on.** MERT (music-focused) and Whisper's encoder (speech-focused) agree on almost nothing in training objective, but both process audio frame-by-frame and both preserve fine pitch/timing detail well. CLAP, which compresses a whole clip into one vector for caption-matching, loses that detail regardless of training data.
