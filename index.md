---
layout: home
author_profile: true
---

Hi, I'm Rupali. I work on language models - text, audio, and video - and mostly ask the same three questions: how trustworthy their reasoning is, whether they can perceive things humans take for granted, and whether they hold up under social pressure.

## Projects

<div class="project-grid">

  <div class="project-card">
    <div class="project-card__icon">🩺</div>
    <h3>Trustworthy Medical QA</h3>
    <p>Fine-tuning language models to answer medical questions with explanations that are actually faithful to the evidence, not just plausible-sounding.</p>
    <div class="project-card__stats">
      <span>5 models</span>
      <span>SFT · DPO · GRPO</span>
      <span>PubMedQA</span>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__icon">🎵</div>
    <h3>How Much Do Models Understand Music?</h3>
    <p>Testing whether audio-language models can actually hear pitch, key, tempo, and meter - or are just guessing from text priors. Playable audio, real model answers.</p>
    <div class="project-card__stats">
      <span>6 models</span>
      <span>13 tasks</span>
      <span>2,063 questions</span>
    </div>
    <div class="project-card__links">
      <a href="{% post_url 2026-07-22-how-much-do-models-understand-music %}">Listen for yourself →</a>
      <a href="https://github.com/rupalirajesh/music-understanding">GitHub ↗</a>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__icon">🎥</div>
    <h3>Sycophancy Resistance</h3>
    <p>Training a video-language model to hold its ground on what it actually sees, even when a user confidently insists otherwise - while still accepting genuine corrections.</p>
    <div class="project-card__stats">
      <span>Qwen2.5-VL</span>
      <span>10K+ pairs</span>
      <span>DPO</span>
    </div>
    <div class="project-card__links">
      <a href="https://github.com/rupalirajesh/sycophancy_VidLMs">GitHub ↗</a>
    </div>
  </div>

</div>
