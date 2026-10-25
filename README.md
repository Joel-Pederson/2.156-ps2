# 2.156 — ps2

October 2026. MIT. 2.156

Fatak Borhani, Joel Pederson, & Leif Akerley

**PS2 objective:** Challenge Problem 2 — submit 100 distinct 7x7 city-zoning grids that all four
advisors rate at least 0.75, maximizing the diversity (mean normalized Hamming distance to the
10 nearest valid neighbours) of the valid set. Due Nov 13, 11:59pm.

Course repo (data, sprites, `utils_public.py`): <https://github.com/decode-mit/2155-CP2-2026>

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
./setup_data.sh                         # clones the course repo into data/cp2-repo (~70 MB, gitignored)
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate ps2
python -m ipykernel install --user --name ps2 --display-name "Python (ps2)"
```

Then open `starter_notebook.ipynb` and select the `Python (ps2)` kernel. Run all — it should go
end to end with no edits.

`environment.yml` covers the course `requirements.txt` plus `pytorch` and `lightgbm` for surrogate
models past the linear baseline. If we never use one of those, drop it and
`mamba env update -f environment.yml --prune`.

Torch device is picked at runtime — Apple Silicon gets MPS, everyone else CPU/CUDA:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

## Layout

| Path | In git? | What |
| --- | --- | --- |
| `starter_notebook.ipynb` | yes | The assignment. Commit **with** outputs. |
| `setup_data.sh` | yes | Fetches the course data. Re-run to update. |
| `submissions/` | yes | The `.npy` files we submit — 4.9 KB each, worth keeping. |
| `data/cp2-repo/` | **no** | Course clone: 500k grids, advisor scores, sprites, `utils_public.py`. |

The notebook's first cell is a bootstrap: on Colab it clones the course repo into `/content`,
locally it uses `data/cp2-repo`. Either way it `chdir`s there, because `utils_public.py` resolves
`datasets/...` and `assets/...` relative to the working directory. It also sets `SUBMISSION_DIR`
to this repo's `submissions/`, so saved submissions land in git rather than in the ignored clone.

Don't commit the course data — it is big, it is not ours, and `setup_data.sh` re-fetches it in
one command.

## The problem, briefly

- 500,000 candidate grids, 7x7, each cell one of 5 districts (Residential 0, Industrial 1,
  Commercial 2, Park 3, Office 4).
- 4 advisors (Wellness, Tax, Transportation, Business) have each scored 5,000 of them in [0, 1].
  Everything else is NaN — that is 1% labelled.
- A grid is **valid** only if *all four* advisors rate it >= 0.75. Invalid grids score nothing and
  are not counted as anyone's neighbour, so a confident design beats a risky one.
- Score = (sum over valid grids of the mean distance to its 10 nearest valid neighbours) / 100,
  so it is capped by the fraction of our 100 that are valid.
- We cannot score ourselves — the staff run the advisors. **5 submissions per day per group**, so
  coordinate before burning one.

Submissions do not have to come from the provided dataset, and the starter notebook explicitly
suggests generating new grids instead of picking existing ones.

## Working together

Notebooks merge badly. To avoid three-way conflicts on cell IDs and outputs:

- Say in chat which section you're taking before you start editing.
- Pull before you edit, push as soon as you're done — don't sit on a dirty notebook.
- Install nbdime once, or the merge driver this repo's `.gitattributes` asks for won't exist:
  ```bash
  mamba install -n ps2 nbdime && nbdime config-git --enable
  ```
  (it is already in `environment.yml`, so a fresh env just needs `nbdime config-git --enable`.)
- Notebooks are committed **with outputs** — the submission needs them.
- If the work grows past one notebook, move the heavy lifting into `.py` modules and keep notebooks
  for looking at results (this is what ps1 ended up doing, and it made merges far less painful).

## Deliverables (each of us submits individually)

- a single `.npy` file: 100x7x7 ints in 0..4
- a zip of the code
- a written reflection report, PDF
