---
layout: single
title: "Research"
permalink: /research/
author_profile: true
---

## Interests

I'm broadly interested in **how language and multimodal models represent the world
internally**, and where that internal representation diverges from what the model can
say or do — the gap between "the information is in there" and "the model uses it."
My current work applies that question to **audio-language models and music**: models
that take audio and text and answer questions about what they hear.

I like problems where a negative or messy result is still a real result — a model
scoring well on a benchmark isn't evidence it's doing the thing you think it's doing,
and figuring out *which* it's doing usually takes more than an accuracy number.

## Projects

### [Where Music Understanding Breaks in Audio-Language Models](/research/music-understanding/)
*Ongoing, 2026.*

A controlled study of six audio-language models — Qwen2-Audio-7B, Qwen2.5-Omni-7B,
Qwen3-Omni-30B, Audio Flamingo 3, Music Flamingo, and Gemini-2.5-Pro — across a
battery of pitch, tempo, key, meter, and instrument-identification tasks, designed
to separate genuine listening from text-prior guessing. Combines behavioral testing,
linear probes on the models' audio encoders, and attention analysis to localize
*where* in the pipeline (encoder, alignment, or language model) each failure lives.

[Read the full writeup →](/research/music-understanding/)

<small>Code: <a href="https://github.com/rupalirajesh/music-understanding">github.com/rupalirajesh/music-understanding</a></small>
