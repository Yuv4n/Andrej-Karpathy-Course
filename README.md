# Micrograd course exercises

My notebooks from Andrej Karpathy's neural-network course. I worked through scalar automatic differentiation and built up a small MLP using `Value` objects. The exercise notebook contains analytic derivatives, finite differences and a softmax/log-loss gradient check.

`Value.backward()` builds a topological order and applies each operation's local derivative in reverse. The implementation notebook also compares a neuron with PyTorch and uses Graphviz to draw the computation graph. The architecture follows the course; these are learning exercises, not original research.

## Open a notebook

From this folder, in a Python environment with Jupyter:

```sh
python3 -m pip install jupyter numpy matplotlib graphviz torch
jupyter notebook micrograd_exercises.ipynb
```

Start with the exercises. The implementation notebook is `micrograd_implementation.ipynb`. Rendering its graph needs the Graphviz system executable (`dot`); the saved Windows PATH line does not install it.

The implementation notebook runs top to bottom: the data cell now precedes training, the duplicated cells are removed, the random seed is fixed and the training loop stops after 1,000 steps. Saved loss values are training loss on four examples, not held-out performance. Rendering the graph needs Graphviz's `dot`.

Course source: [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero).
