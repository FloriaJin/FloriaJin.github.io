---
layout: page
title: Targeted Latent Adversarial Training — Reproduction
description: Laptop-scale reproduction of latent-space adversarial robustness for LLMs.
img:
importance: 2
category: work
github: https://github.com/FloriaJin/latent-adversarial-training-reproduction
---

Reproduction of **Targeted Latent Adversarial Training** ([arXiv:2407.15549](https://arxiv.org/abs/2407.15549)). I contributed code fixes grounded in the paper's future-work section (hidden-size handling, device-agnostic execution) and ran a laptop-scale GPT-2 latent-attack study that reproduces the reported layer × epsilon sensitivity of the defense.

Original implementation: [aengusl/latent-adversarial-training](https://github.com/aengusl/latent-adversarial-training).

**Stack:** PyTorch · Transformers · Jupyter
