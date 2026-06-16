---
layout: page
title: Safe-RLHF & Latent Adversarial Training
description: Helpfulness vs. harmlessness as separate signals; robustness to persistent harmful behaviors.
img:
importance: 3
category: research
---

Executed the full **Safe-RLHF** sequence (SFT, reward model, cost model, PPO-Lag) with DeepSpeed, Transformers, and W&B, studying helpfulness and harmlessness as separate optimization signals connected through KL control and a Lagrangian safety constraint.

I also traced **Latent Adversarial Training**, perturbing the residual stream (via PyTorch, PEFT, TransformerLens) to study robustness against jailbreaks, backdoors, and unlearning—working the red-team side of the alignment pipeline.

**Stack:** DeepSpeed · Transformers · TransformerLens · PEFT
