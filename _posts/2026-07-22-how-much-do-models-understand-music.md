---
layout: single
title: "How Much Do Models Understand Music?"
date: 2026-07-22
author: Rupali Rajesh
excerpt: "Testing AudioLMs on hearing tasks."
---

<div class="stat-row">
  <div class="stat-tile"><span class="stat-num">6</span><span class="stat-label">models tested</span></div>
  <div class="stat-tile"><span class="stat-num">13</span><span class="stat-label">listening tasks</span></div>
  <div class="stat-tile"><span class="stat-num">5</span><span class="stat-label">examples below</span></div>
</div>

Models tested: Qwen2-Audio-7B, Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B, Audio Flamingo 3,
Music Flamingo, Gemini-2.5-Pro.

<div class="audio-example">
  <h4>1. Instrument identification</h4>
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
</div>

<div class="audio-example">
  <h4>2. Octave placement</h4>
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
</div>

<div class="audio-example">
  <h4>3. Beats per bar</h4>
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
</div>

<div class="audio-example">
  <h4>4. Key identification</h4>
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
</div>

<div class="audio-example">
  <h4>5. Fixing pitch perception</h4>
  <table class="model-answer-table">
    <thead><tr><th>Task</th><th>Before</th><th>After</th></tr></thead>
    <tbody>
      <tr><td>Cents discrimination (how far off-pitch)</td><td>0.55</td><td class="ok">0.94</td></tr>
      <tr><td>Tuning judgment (in-tune or not)</td><td>0.53</td><td class="ok">0.89</td></tr>
    </tbody>
  </table>
  <p class="audio-example__caption">Fine-tuned with a graph depicting the audio pitch.</p>
</div>
