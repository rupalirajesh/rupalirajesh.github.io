---
layout: home
author_profile: true
---

Hi, I'm Rupali. I work on language models - text, audio, and video - and mostly ask the same three questions: how trustworthy their reasoning is, whether they can perceive things humans take for granted, and whether they hold up under social pressure.

## Projects

<div class="project-grid">

  <div class="project-card">
    <div class="project-card__icon">🩺</div>
    <h3>Teaching Trustworthiness</h3>
    <p>Fine-tuning language models to give more trustworthy explanations - improving correctness, consistency, and faithfulness in stages.</p>
    <div class="project-card__stats">
      <span>5 models</span>
      <span>SFT · DPO · GRPO</span>
      <span>PubMedQA</span>
    </div>
  </div>

  <div class="project-card">
    <div class="project-card__icon">🎵</div>
    <h3>How Much Do Models Understand Music?</h3>
    <p>Testing whether audio-language models actually understand music - across tests in pitch, key, tempo, harmony, meter, and more.</p>
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
    <h3>Mitigating Sycophancy in Video LMs</h3>
    <p>Fixing the sycophancy RLHF trains into video-language models, and getting them to actually attend to what's in the video instead of just what the prompt says.</p>
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
