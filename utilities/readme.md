# TensorFlow Graph & Autodiff Static Scanner

A fast static linter that catches **GPU-graph incompatibilities**, **autodiff gotchas**, and optional **broadcasting/rank-drop** issues in TensorFlow projects. It follows calls from compiled functions into helpers and reports findings with **call chains**, **colorized CLI**, and **JSON** output.

---

## Why use it?

- Spot silent performance killers (`x.numpy()`, `print()` in graphs, `python and/or` on tensors).
- Catch gradient blockers (`tf.py_function`, `tf.cast(..., tf.int32)`, discrete ops on grad path).
- Optionally warn on shape foot-guns like `(N,)` vs `(N,1)`.

---

## Install

No package install required; just drop `utilities/scan_tf_graph_readiness.py` into your repo.

Optional marker decorators (no-ops) — add to `utilities/decorators.py`:

```python
def requires_grad(fn): return fn
def no_grad_ok(fn): return fn
def broadcast_ok(fn): return fn
