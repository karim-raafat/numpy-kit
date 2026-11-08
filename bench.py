import timeit

import numpy as np

from numpy_kit.ops import (
  pairwise,
  pairwise_loop,
  pairwise_v2,
  softmax,
  softmax_loop,
  standardize,
  standardize_loop,
)

rng = np.random.default_rng(42)


sf_a = rng.standard_normal((1000,100))

ps_a = rng.standard_normal((200,64))
ps_b = rng.standard_normal((200,64))

std_a = rng.standard_normal((1000,50))


def best_ms (func, *args):
  times = timeit.repeat(lambda: func(*args), number = 1, repeat = 5)
  best = min(times)

  return best * 1000



rows = [
    ("softmax",     softmax_loop,     softmax,     (sf_a,)),
    ("pairwise v1", pairwise_loop,    pairwise,    (ps_a, ps_b)),
    ("pairwise v2", pairwise_loop,    pairwise_v2,    (ps_a, ps_b)),
    ("standardize", standardize_loop,    standardize,    (std_a,)),
]

print(f"{'function':<15}{'loop (ms)':>12}{'vectorized (ms)':>16}{'speedup':>10}")
for name, loop_fn, vecfn, inputs in rows:
  loop_ms = best_ms(loop_fn, *inputs)
  vec_ms = best_ms(vecfn, *inputs)
  speedup = loop_ms / vec_ms
  print(f"{name:<15}{loop_ms:>12.2f}{vec_ms:>16.2f}{speedup:>10.0f}×")







