---
layout: page
title: Price Elasticity via Double Machine Learning
description: Debiased causal estimation of price elasticity with cross-fitting.
img:
importance: 1
category: applied
---

Built a **Double ML** pipeline to estimate price elasticity: engineered product, temporal, and stock confounders, residualized log-price and log-quantity with RandomForest models via cross-fitting, then estimated debiased causal elasticity through OLS on the residuals.

**Stack:** scikit-learn · pandas · RandomForest · Causal Inference
