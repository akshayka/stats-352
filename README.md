# Stats 352: Fixing Python notebooks with marimo

> Slides: https://www.figma.com/slides/EtqHkcMZbqieKGUCUirVaw/2025.04.16-marimo---Stats-352?t=betZNE5LtMOhpbpe-1

marimo is open source:

* repo: https://github.com/marimo-team/marimo
* docs: https://docs.marimo.io
* landing page: https://marimo.io

## Set up

marimo can be installed from PyPI into your virtual environments, just like
Jupyter. But for convenience, we'll use marimo through `uv`.

Install the `uv` package manager:

- https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Test marimo with:

```bash
uvx marimo tutorial intro
```


## Example notebooks

Example notebooks are in the notebook directory. These have their
package requirements serialized in them, and marimo can [install
these requirements for you](https://docs.marimo.io/guides/package_reproducibility/) if you have `uv` installed.

Run notebooks with

```bash
uvx marimo edit my_notebook.py
```

Run a notebook as a script with:

```bash
uv run my_notebook.py
```

Try this on the exploring languages notebook, in the `notebooks` directory:

```bash
uv run exploring_languages.py Python Rust --p 4
```

## Sharing your work on GitHub Pages

marimo notebooks can be published with full interactivity to GitHub pages,
for free. These notebooks run in the browser and can use most packages
from the scientific computing stack, though modern machine learning
libraries like PyTorch and TensorFlow do not yet work.

Get started with this template: https://github.com/marimo-team/marimo-gh-pages-template
