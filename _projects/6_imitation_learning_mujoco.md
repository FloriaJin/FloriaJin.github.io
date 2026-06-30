---
layout: page
title: Imitation Learning for Continuous Control
description: Behavioral Cloning vs. DAgger on MuJoCo locomotion, and when cloning breaks.
img:
importance: 6
category: work
github: https://github.com/FloriaJin/imitation-learning-mujoco
---

An implementation and empirical study of two imitation learning algorithms—Behavioral Cloning (BC) and DAgger (Dataset Aggregation)—on high-dimensional MuJoCo locomotion tasks. Neural-network policies are trained to imitate expert demonstrations, and the analysis characterizes when naive cloning succeeds, when compounding errors push the agent into states the expert never visited, and how interactive expert relabelling closes the distribution-shift gap.

**Stack:** Python · PyTorch · MuJoCo · Gym
