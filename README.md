# 3DGS for Earth Observation — PhD repo

Canopy-aware + spectral 3D Gaussian Splatting for satellite imagery (UAH CS PhD).
Owner: Navaneeth. See the planning docs for full reasoning:
`HANDOFF.md.docx`, `STATUS.md.docx`, `PhD_Research_Plan_3DGS_EarthObservation.md.docx`,
`Papers_1_and_2_DeepDive.md.docx`.

**Read `STATUS.md` first every session. Update it last, then commit + push.**

---

## The immediate goal (gates everything)

Reproduce **EOGS** on DFC2019 tiles, produce a DSM, and match its reported
mean-absolute-error vs. the provided lidar. When that number matches, the
pipeline is real and every new contribution (lidar fusion, spectral bands)
is an addition to it. Nothing new starts until this passes.

Good news from setup research: the EOGS repo
([github.com/mezzelfo/EOGS](https://github.com/mezzelfo/EOGS)) ships a release
`data.zip` containing the exact pre-processed JAX/IARPA tiles **with DSM ground
truth**, plus a one-command reproduction. So the milestone needs **no IEEE
DataPort download and no Earthdata account** — those come later for Paper 1's
GEDI/ICESat-2/HLS fusion.

---

## How the two machines work together

- **Laptop** = planning, reviewing results, writing the next experiment, committing it.
- **4090 PC** = pulling data, building the env, running training, writing results back.
- **Source of truth** = this git repo + `STATUS.md`. Chat history/sandbox do **not**
  sync between machines; only the repo does.
- Keep code + configs + small results in git. Keep big data/checkpoints **local to
  the 4090** (gitignored) and track them in `data/manifest.csv`.

---

## Repo layout

```
PhD work/
├── README.md                  # this file
├── STATUS.md                  # living lab notebook — read first, update last
├── .gitignore                 # excludes big data, checkpoints, the EOGS clone
├── environment.yml            # conda env reference (real build via scripts/01)
├── configs/
│   └── milestone.env          # central paths / URLs / scene list — sourced by all scripts
├── scripts/
│   ├── init_git.sh            # ONE TIME: clean repo on 'main' + first commit (+remote)
│   ├── 00_check_gpu.sh        # nvidia-smi + torch.cuda.is_available()
│   ├── 01_setup_env.sh        # conda env + clone EOGS --recursive + install + CUDA kernels
│   ├── 02_earthdata_auth.py   # earthaccess login test (Paper 1, not needed for milestone)
│   ├── 03_get_eogs_data.sh    # download EOGS release data.zip + unzip into the clone
│   ├── 04_prep_cameras.sh     # run EOGS to_affine.py for the milestone scenes
│   ├── 05_run_eogs.sh         # bash train.sh reproduceMain  (the milestone)
│   ├── 06_eval_dsm_mae.py     # standalone DSM MAE/RMSE/completeness vs truth
│   └── run_milestone.sh       # orchestrates 03 -> 04 -> 05 (+ optional 06)
├── data/
│   ├── README.md              # data policy (never sync big data)
│   └── manifest.csv           # what's downloaded where on the 4090
└── notes/
    └── session_template.md
```

The third-party EOGS code is cloned **outside** the tracked tree (default
`~/eogs-src/EOGS`, gitignored) so this repo stays small and syncable.

---

## Quickstart on the 4090 (WSL2 Ubuntu)

```bash
cd "/path/to/PhD work"           # your WSL2 path to this folder
bash scripts/init_git.sh         # ONE TIME: clean repo on 'main' + first commit
#   add your private GitHub remote: bash scripts/init_git.sh git@github.com:<you>/<repo>.git
bash scripts/00_check_gpu.sh     # confirm GPU + driver
bash scripts/01_setup_env.sh     # build conda env 'eogs', clone+install EOGS (~10-20 min)
conda activate eogs
bash scripts/run_milestone.sh    # download data -> prep cameras -> reproduce -> MAE
```

Then record the MAE numbers + any gotchas in `STATUS.md`, commit, and push.
**Stop and report the MAE before building anything new.**

See `scripts/` headers and the troubleshooting notes in `01_setup_env.sh`
for CUDA-toolkit / submodule build issues on WSL2.

---

## Edit these scripts before first run

Open `configs/milestone.env` and confirm:
- `EOGS_DIR` — where to clone EOGS (default `~/eogs-src/EOGS`)
- `CONDA_ENV` — conda env name (default `eogs`)
- `TORCH_INDEX_URL` — CUDA wheel index (default cu121; bump if your driver needs cu124)
- `MILESTONE_SCENES` — which tiles to reproduce (default the 4 JAX scenes)

---

## Reproducibility contract (for the advisor / fresh machines)

`scripts/01_setup_env.sh` is designed to be the **single self-installing entry point**.
On a clean WSL2/Ubuntu box it installs everything needed end-to-end:

- Miniconda (if `conda` is missing) → `~/miniconda3`
- accepts the Anaconda default-channel Terms of Service
- system build tools via apt (`build-essential`, `cmake`, GDAL/TIFF/PNG/JPEG/FFTW dev libs)
  — required to compile EOGS's native deps (`iio`, `srtm4`, `plyflatten`)
- the `eogs` conda env + PyTorch (CUDA wheels) + EOGS + geospatial tooling
- the 3DGS CUDA kernels (when a CUDA toolkit / `nvcc` is present)

**Policy:** every setup error we ever hit gets fixed *in the script*, never left as a
manual one-off. If you find a gap on your machine, tell us the error and it gets folded in,
so the next person runs one command and it just works.
