---
layout: page
title: Circuit Breakers — Representation Engineering
description: Altering harmful hidden representations while retaining benign behavior.
img:
importance: 2
category: research
---

Trained circuit-breaker adapters by wiring `CustomTrainer`, `compute_loss`, and `CircuitBreakerDataset` with Transformers, PEFT LoRA, PyTorch, and DeepSpeed, to alter harmful hidden representations while keeping benign behavior intact. Mapped the full workflow—from retain data and circuit-breaker prompts through tokenization masks, hidden-state loss, and LoRA merge—to contrast representation intervention with refusal-only alignment.

**Stack:** Transformers · PEFT/LoRA · DeepSpeed · PyTorch
