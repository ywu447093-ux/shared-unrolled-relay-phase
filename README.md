# Shared-weight unrolled relay-screen phase retrieval

Code accompanying **Shared-weight unrolled phase retrieval with transmission priors for simulated relay-screen imaging** by Yifan Wu and Yang Wang.

This is the R-V2 computational source snapshot (`rv2-code-1.0.0`).
The exact manuscript identity is recorded in `PAPER_VERSION.json`; scientific
source hashes and their archived origins are recorded in `SOURCE_INDEX.json`.
The original early V12 package alone is not the experiment suite used in R-V2.

## Release contents

The release includes the shared reconstruction,1/2/3/4/6/8-stage and attention
ablations, partial sharing, calibrated direct networks, adapted GS/HIO/HIO–ER/
NCPP comparisons, retrospective and regenerated object conditions, balanced
absorption pairing, matched additional training, FourierGSNet adaptation,
same-network no-diffraction control, metric/interval analysis and plotting.

Only source code and implementation documentation are published. Model
weights, simulation caches, raw QPM images, predictions, numerical result
tables and manuscript files are not included. Cloning this repository does
not by itself reproduce trained-checkpoint accuracy. Full regeneration and
training require the original frozen assets described below.

## Environment

The audited environment is Windows,Python3.12 and PyTorch2.9.1+CUDA13.0.
Exact audited versions are in `ENVIRONMENT_AUDIT.json`; `requirements.txt`
provides an installation with compatible NumPy/OpenCV package constraints.
That dependency selection has not been used to regenerate paper results.
The numerical scripts use BF16 neural operations
with FP32 propagation and metrics. Replaying the recorded evaluation batch
boundaries matters: most neural evaluations use batch4; FourierGSNet uses2.
Singleton mixed-precision evaluation can differ locally from batched output.
Some frozen asset-generation code uses the Windows `msvcrt` locking module.

```text
python -m pip install -r requirements.txt
python verify_release.py
```

`verify_release.py` checks the preserved sources and expected source-only
contents. With PyTorch installed, `--smoke` also tests a synthetic forward
pass; it does not load any reference label or claim reproduction of paper
accuracy. The optional SiSPRNet source must be obtained separately as described
in `THIRD_PARTY_NOTICES.md`.

## Assets and execution

The frozen scripts preserve their research-era paths and computations. Use
the provided explicit-path launchers instead of relying on those defaults.
Let `ASSETS` be the original development-asset root containing:

```text
outputs/R1_controlled_experiments_v2/training_cache.pt
outputs/R1_controlled_experiments_v2/source_split.json
outputs/compact_ncpp_v6/augmented_cache.pt
outputs/compact_ncpp_v6/augmentation_manifest.json
work/real_cell_dataset_v11/sps/train_*.npy
work/real_cell_dataset_v11/manifest.csv
```

These files are external assets, not included in this source-only release.
The QPM morphology source is https://doi.org/10.5281/zenodo.5153251.
Original TIFF values are in pg/um^2; the simulations assign new phase scale
and sampling. Real relay-screen measurements are not part of this study.

```text
python work/experiments/portable.py --mode train --assets-root ASSETS --arms stage4 --epochs 40 --seed 101
python work/experiments/portable.py --mode calibrated --assets-root ASSETS
python work/experiments/portable.py --mode stability --assets-root ASSETS
```

The complete classical runner subsequently invokes the calibrated, lower-rate
and PhysenNet runners. Read its entry point before launching a long run.
Its actual GS/HIO/HIO–ER/NCPP object and measurement constraints are implemented
in `work/experiments/classical.py`; the label `NCPP adapted` refers to this
study's control, not an equivalence-certified original-author implementation.

Additional-training and evaluation launchers:

```text
python work/experiments/evidence_reproduce.py --mode prepare-corrective --assets-root ASSETS --raw-qpm-root RAW_QPM --run-dir NEW_EMPTY_DIRECTORY
python work/experiments/evidence_reproduce.py --mode train-corrective --assets-root ASSETS --raw-qpm-root RAW_QPM --run-dir ANOTHER_NEW_EMPTY_DIRECTORY
```

The launchers require original selected checkpoints for the additional-training
arms and the corresponding generated observations for evaluation. They do not
download unpublished weights or replace missing assets with other results.
Training a new model produces a new result; it must not be silently substituted
for a retained paper checkpoint. `evidence_infer.py` documents checkpoint
loading and excludes reference fields from the forward inputs.

## Code map

| Study component | Entry point |
|---|---|
| Main model, depth, attention and sharing | `work/experiments/suite.py` |
| Calibrated direct baselines | `work/experiments/calibrated.py` |
| GS/HIO/HIO–ER/NCPP adaptations | `work/experiments/classical.py` |
| Retrospective and object-condition simulation | `work/experiments/reviewer_validation.py` |
| Fixed one-to-one absorption pairing | `work/experiments/balanced_pairing.py` |
| Matched additional training | `work/experiments/reviewer_training.py` |
| FourierGSNet and no-diffraction controls | `work/experiments/fouriergs_paired.py`, `no_diffraction_train.py` |
| Phase errors, conditional intervals | `work/reviewer_phase_diagnostics.py`, `work/experiments/evidence_analysis.py` |
| Final plot construction | `work/figure_refinement/build_figures.py` |

The later independent-SPS diagnostic study and its paper-pseudocode NCPP
transcription are outside this manuscript snapshot. The current release does
not substitute those methods for the adapted baseline used in R-V2.

## Source provenance

The listed scientific files are copied byte-for-byte from the retained
research sources. Later manuscript language, title and figure-label edits did
not retrain those models. The six numerical main-text tables in the author-
specified R-V2 file match the audited result version. These facts establish
source provenance; they do not make the code-only repository a weights/data
release or a guarantee of acceptance by a journal.

## Third-party sources and citation

Read `THIRD_PARTY_NOTICES.md` before using the literature baselines. The
SiSPRNet source is intentionally not redistributed; FourierGSNet retains its
MIT notice. `CITATION.cff` identifies the associated authors and source version.
The manuscript has not been assigned a DOI in this code snapshot.
