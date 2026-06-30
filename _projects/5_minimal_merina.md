---
layout: page
title: Minimal MERINA+
description: A compact, readable reimplementation of meta-RL adaptive bitrate streaming.
img:
importance: 5
category: work
github: https://github.com/FloriaJin/Minimal-MERINA-reproduction
---

A compact, self-contained reproduction of the core ideas in **MERINA+** ([IEEE TCSVT 2025](https://ieeexplore.ieee.org/document/11119689)), a meta-reinforcement-learning approach to adaptive bitrate (ABR) video streaming. The goal is to make the method easy to read, run, and reason about: a pure-Python simulator, three small networks, and a PPO + variational-information-bottleneck training loop that runs in a few minutes on a free Colab GPU or a laptop CPU.

This is a teaching-oriented reimplementation focused on the algorithmic core rather than the paper's exact benchmark numbers. Original implementation: [confiwent/merina-plus](https://github.com/confiwent/merina-plus).

**Stack:** Python · PyTorch · PPO
