# numpy-kit

A small NumPy practice library that implements common numerical operations **twice**: once with plain Python loops and once fully vectorized with NumPy. It then benchmarks the two versions against each other to show how much faster vectorized code is.

Each loop version is the readable reference. The tests check that every vectorized version gives the same results as its loop version.

## What's inside

All functions live in `src/numpy_kit/ops.py`.

| Operation | Loop version | Vectorized version(s) | Description |
|---|---|---|---|
| Softmax | `softmax_loop`, `softmax_loop_1d_loop` | `softmax` | Numerically stable softmax (subtracts the row max) over a 1D array or each row of a 2D array. |
| Pairwise distance | `pairwise_loop` | `pairwise`, `pairwise_v2` | Euclidean distance matrix between rows of `a` (n×d) and `b` (m×d), returning n×m. |
| Standardize | `standardize_loop` | `standardize` | Column-wise z-score normalization. Returns `(z, means, stds)`; constant columns map to 0 instead of NaN. |

### The two pairwise versions

- **`pairwise` (v1)** uses broadcasting: `a[:, None, :] - b[None, :, :]`, then squares, sums and takes the square root. It is simple, but it builds an intermediate array of size n×m×d.
- **`pairwise_v2`** uses the identity `‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·bᵀ`, so most of the work becomes one matrix multiply (`a @ b.T`). It never builds the 3D array, and it clips small negative values caused by floating-point error to 0 before taking the square root.

## Benchmark results

`bench.py` times each loop/vectorized pair. It takes the best of 5 runs, using random inputs with a fixed seed of 42:

| Function | Input shape | Loop (ms) | Vectorized (ms) | Speedup |
|---|---|---:|---:|---:|
| softmax | (1000, 100) | 25.14 | 0.23 | **108×** |
| pairwise v1 | (200, 64) vs (200, 64) | 348.42 | 1.66 | **210×** |
| pairwise v2 | (200, 64) vs (200, 64) | 338.32 | 0.08 | **4474×** |
| standardize | (1000, 50) | 15.83 | 0.09 | **177×** |

The matrix-multiply version, `pairwise_v2`, is about 20× faster than the broadcasting version, `pairwise`, and over 4000× faster than the pure-Python loop. Exact timings depend on your machine.

## Project structure

```
numpy-kit/
├── src/numpy_kit/
│   ├── __init__.py
│   ├── ops.py          # loop + vectorized implementations
│   └── py.typed
├── tests/
│   └── test_ops.py     # correctness tests and loop vs vectorized equivalence tests
├── bench.py            # benchmark script that prints the table above
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended)
- NumPy ≥ 2.4.6. Dev tools: pytest and ruff.

## Getting started

```bash
# Install dependencies (creates .venv)
uv sync

# Run the tests
uv run pytest

# Run the benchmark
uv run python bench.py

# Lint
uv run ruff check .
```

## Usage

```python
import numpy as np
from numpy_kit.ops import softmax, pairwise_v2, standardize

x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
softmax(x)                      # each row sums to 1

a = np.random.rand(5, 3)
b = np.random.rand(7, 3)
pairwise_v2(a, b).shape         # (5, 7)

z, means, stds = standardize(np.random.rand(100, 4))
```

## Tests

The test suite in `tests/test_ops.py` (33 tests) covers:

- Correctness on known inputs (for example, softmax sums to 1 and hand-computed distance matrices are reproduced)
- Numerical stability (large softmax inputs give no NaN, constant columns in standardize give no NaN)
- Input validation (empty arrays, wrong number of dimensions, mismatched feature counts)
- The input array is not modified
- Vectorized and loop results match across several input shapes
