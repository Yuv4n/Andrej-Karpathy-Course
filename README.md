# Micrograd course notebooks

Notebooks following Andrej Karpathy's [neural-network course](https://github.com/karpathy/nn-zero-to-hero): scalar automatic differentiation and a small MLP built from `Value` objects. These are learning exercises, not original research.

- **`micrograd_exercises.ipynb`:** analytic and finite-difference derivatives, and a softmax/log-loss gradient check.
- **`micrograd_implementation.ipynb`:** `Value.backward()` via topological order, an MLP trained on four examples (seeded, 1,000-step cap), and a comparison with PyTorch.

## Run

```sh
python3 -m pip install jupyter numpy matplotlib graphviz torch
jupyter notebook micrograd_exercises.ipynb
```

Rendering the computation graph needs the Graphviz `dot` executable.

## Limits

Loss values are training loss on four examples, not held-out performance.
