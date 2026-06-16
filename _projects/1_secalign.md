---
layout: page
title: SecAlign — Prompt-Injection Defense
description: Reproducing preference-optimization defenses against prompt injection.
img:
importance: 1
category: research
---

Reproduced **SecAlign**, which defends LLMs against prompt injection by teaching the model to prefer legitimate instructions over injected ones. I ran the full training pipeline (`align.py`, `train.py`, `run.py`) with PyTorch, Transformers, DPO, and LoRA, and constructed preference data from prompt-injected inputs paired with secure vs. insecure outputs—comparing preference optimization against StruQ-style defensive SFT.

**Stack:** PyTorch · Transformers · DPO · LoRA
