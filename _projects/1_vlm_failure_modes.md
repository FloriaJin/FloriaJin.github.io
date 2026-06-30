---
layout: page
title: VLM Failure Mode Analysis
description: Do vision-language models trust the image, or follow misleading language priors?
img:
importance: 1
category: work
github: https://github.com/FloriaJin/vlm-failure-mode-analysis
---

A small-scale, reproducible study of failure modes in open-source vision-language models (VLMs). Rather than training a new model, the project constructs controlled synthetic image–text cases to test whether a VLM relies on visual evidence or follows misleading language priors when the two conflict.

The benchmark covers four interpretable categories—false presupposition / object absence, attribute conflicts, counting traps, and spatial-relation errors—and measures how often the model accepts a false premise embedded in the prompt instead of answering from the image. Evaluated on Qwen-VL.

**Stack:** Python · Transformers · Qwen-VL
