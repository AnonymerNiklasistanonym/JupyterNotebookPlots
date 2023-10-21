# JupyterNotebookPlots

A collection of Jupyter Notebook plots

## How to use?

Some are just random plots that can be copied directly but some are implemented in a library function.
you can just copy the `lib` directory (and rename it however you want) and then import it into your jupyter notebook via:

```python
# Import from relative external file (this example would be a
# `root/python/lib/plots.py` dir while the file is in `root/dir/file.ipynb`)
import os
import sys

module_path = os.path.abspath(os.path.join("..", "python"))
if module_path not in sys.path:
    sys.path.append(module_path)

# Import it like this if it's in the same directory as the notebook/file
from lib.plots import (
    plot_random_variable_distribution,
    plot_random_variable_distribution_function,
    plot_random_variable_constant_distribution_function,
)
```

## Jupyter Notebooks

- [`plots.ipynb`](plots.ipynb): Plots regarding statistics
