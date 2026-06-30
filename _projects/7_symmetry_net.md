---
layout: page
title: Mirror-Symmetry Detection Net
description: A from-scratch NumPy reproduction of the classic 1986 backprop symmetry network.
img:
importance: 1
category: fun
github: https://github.com/FloriaJin/symmetry-net
---

A from-scratch reproduction of the mirror-symmetry detection network from Rumelhart, Hinton & Williams (1986, Fig. 1). The 6→2→1 sigmoid network is trained with batch gradient descent plus momentum to decide whether a 6-bit binary pattern is left–right mirror-symmetric—reproducing the textbook result that just two hidden units suffice. Implemented in pure NumPy, no autograd, to make the backprop math explicit.

**Stack:** Python · NumPy
