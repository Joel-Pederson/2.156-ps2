# 2.156 — ps2
October 2026. MIT. 2.156

Fatak Borhani, Joel Pederson, & Leif Akerley

**PS2 objective:** _TODO — one line, once the assignment is out._

## Prerequisites

- **A conda installer.** [Miniforge](https://github.com/conda-forge/miniforge) is recommended (it ships
  `mamba` and defaults to conda-forge). Already have Anaconda or Miniconda? That works too — use `conda`
  wherever `mamba` appears below.
  - macOS (recommended): `brew install miniforge`, or download the installer from the Miniforge page and run
    `bash Miniforge3-MacOSX-$(uname -m).sh`
  - Linux: `bash Miniforge3-Linux-$(uname -m).sh`
  - Then `conda init` (restart your terminal) so `conda activate` works.
- **Git**, and access to this repo (ask Joel to add you as a collaborator).
- **VS Code** with the Python and Jupyter extensions (or use JupyterLab, which the env includes).

## Setup

```bash
git clone git@github.com:Joel-Pederson/2.156-ps2.git   # no SSH key? use https://github.com/Joel-Pederson/2.156-ps2.git
cd 2.156-ps2
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate ps2
python -m ipykernel install --user --name ps2 --display-name "Python (ps2)"
```

Then open the notebook and select the `Python (ps2)` kernel.

`environment.yml` is still the course template's default dependency list. It gets updated from the
assignment notebook's `!pip install` lines once the notebook is in. After it changes:
`mamba env update -f environment.yml --prune`.

Torch device is picked at runtime — Apple Silicon gets MPS, everyone else CPU/CUDA:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

## Working together

Notebooks merge badly. To avoid three-way conflicts on cell IDs and outputs:

- Say in chat which section you're taking before you start editing.
- Pull before you edit, push as soon as you're done — don't sit on a dirty notebook.
- Install nbdime once, or the merge driver this repo's `.gitattributes` asks for won't exist:
  ```bash
  mamba install -n ps2 nbdime && nbdime config-git --enable
  ```
- Notebooks are committed **with outputs** — the submission needs them.
- If the work grows past one notebook, move the heavy lifting into `.py` modules and keep notebooks
  for looking at results (this is what ps1 ended up doing, and it made merges far less painful).
