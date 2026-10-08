---
title: Quickstart
description: Deconvolve a reaction time course in a few lines.
sidebar:
  order: 2
---

The curve-resolution workflow, with a runnable example, is in
[Curve resolution with honest uncertainty](../curve-resolution/).

Every example needs float64, switched on once at the top of your script:

```python
import jax

jax.config.update("jax_enable_x64", True)
```
