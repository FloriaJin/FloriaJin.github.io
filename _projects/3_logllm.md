---
layout: page
title: LogLLM — Reproduction
description: Reproducing LLM-based log anomaly detection, with reproducibility and robustness fixes.
img:
importance: 3
category: work
github: https://github.com/FloriaJin/LogLLM-reproduction
---

Reproduction of **LogLLM** ([arXiv:2411.08561](https://arxiv.org/abs/2411.08561)), a framework for log anomaly detection built on large language models. The contribution here is code-only: reproducibility seeding, replacing hard-coded paths with `argparse`, a bugfix in robust answer parsing, and optional quantization / CPU execution so the pipeline runs without a high-end GPU.

Original implementation: [guanwei49/LogLLM](https://github.com/guanwei49/LogLLM).

**Stack:** Python · Transformers
