# The World Begins with Subjectivity Intersection
## From Braid Quantum Gravity to Embodied Experience

Reproduction materials for Satoru Watanabe's published preprint, version 1.

- Paper DOI: https://doi.org/10.5281/zenodo.23284178
- ResearchGate: https://www.researchgate.net/publication/415538936
- Author: Satoru Watanabe, Subjectivity Intersection Emergence Lab (SIEL)
- Contact: satoru@siel.global

This repository reproduces numerical results reported in the paper. It supplies fixed Braid kernels, pseudonymous task-level features, raw-derived EEG feature arrays, original numerical functions, complete retained null/comparator rows, and finite-source records. It distinguishes recomputation from released features, recalibration of retained null outputs, and rebuilding the finite source from its original matrices.

## Quick start

Tested with Python 3.14.6, NumPy 2.4.4, pandas 3.0.2, and SciPy 1.17.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_manifest.py
python scripts/reproduce.py --output results/core
python scripts/reproduce_secondary.py --output results/secondary
python scripts/verify_records.py --output results/records
```

Each output directory must be new: the runners refuse to overwrite prior runs. Run from the repository root. No author-specific filesystem paths, private Git repository, login, or network connection is needed by these commands after dependency installation.

For the longer checks:

```bash
python scripts/reproduce.py --module eeg --full --output results/full-eeg
python scripts/reproduce.py --module finite --full --output results/full-finite
```

`--full` EEG recomputes scores for all 1,000 retained random-kernel ensembles. `--full` finite rebuilds the fixed source matrices, stationary states, spectral projectors, Gram records, and compressed states, then compares the electrical responses and ranks exactly. These are checks of retained calculations, not new experiments.

## Paper-to-code map

| Paper | Command | Reproduction scope |
|---|---|---|
| §3.3, §4.2 | `reproduce.py` | Eight-component fingerprint; 26 own-session and 26 P1-source comparisons; 26/26 topology-positive and 23/26 both-positive scores; 2,000 P1 permutations |
| §4.2 controls | `reproduce.py` | Participant calibration from 52,000 retained null rows; all 676 cyclic shifts; optional 26,000 random-kernel scores |
| §4.3 | `reproduce_secondary.py` | P1 full-window H1-zero signed source profile, including the 1/26 Braid and 12/26 generic source ranks |
| §4.4 | `reproduce_secondary.py` | All 624 receiver/phase/source scores and reported P1 post-midpoint p-value from retained null rows |
| §5.2–5.3, §7.2 | `verify_records.py` | Exact all-eight complementary-purity separation; all 24 directed update edges and two-order histories |
| §5.7–5.8, Appendix E | `reproduce.py --module finite` | Exact current response matrices and rank 3 for all eight sources; full mode rebuilds the source records |
| §6.2 | `reproduce_secondary.py` | Lightlike source/band correspondence and P1 participant-adjusted null calibration |
| §6.2–6.4 | `reproduce_secondary.py` | EEG-only band-group scores; all participant forward-minus-reverse band correlations; retained-null calibration |

See [Reproduction scope](docs/REPRODUCTION_SCOPE.md), [Data dictionary](docs/DATA_DICTIONARY.md), and [Provenance](docs/PROVENANCE.json).

## Acquisition and interpretation

EEG and quantum measurements were obtained simultaneously for the same participant in the same session. The measurement systems had no data linkage or feedback. The 26 units within each session and the midpoint windows are post-acquisition analysis units and windows, not behavioral tasks or hardware synchronization triggers.

The EEG analyses are exploratory analyses of one existing archive. Positive-score counts are descriptive counts. The repository retains random-kernel and cyclic-shift comparisons as well as their multiplicity-adjusted results. It also preserves the original NO-GO decisions for the exclusive three-way EEG band partition and the primary time-window prediction; the secondary observations do not replace those decisions.

The electrical-readout result is conditional mathematics for specified finite-source states and ideal observables. It does not demonstrate implementation in Kv1.2, transmission to EEG, or subjective experience. The embodiment and ontological proposals are hypotheses and interpretations, not additional numerical endpoints certified by this repository.

## Data and source boundaries

No raw EEG waveforms, acquisition timestamps, demographic records, consent forms, credentials, or participant identity mappings are included. Participant labels remain pseudonymous. Only the numerical inputs needed for the reported comparisons and their retained reference outputs are supplied.

The raw-derived arrays were regenerated from hash-verified original inputs during preparation. Default reproduction starts from these released intermediate features. It does not claim to rebuild the original acquisition/preprocessing pipeline from raw EEG. Prior raw data and reproduction materials are cited at https://doi.org/10.5281/zenodo.19624924 and https://github.com/SATORU-SIEL/EEG-Quantum-Correlations. This repository does not relicense those external materials.

`src/lineages/` contains original helper modules for provenance. Some historical standalone drivers retain their historical path assumptions or require upstream raw files. Use the portable commands above; those commands do not invoke these historical standalone entry points.

## Citation and license

Cite the paper version and the repository commit used. See `CITATION.cff`.

Author-created materials in this release are licensed under CC BY 4.0. External sources retain their own terms. Historical manuscript/result identifiers are retained as provenance, not as a claim of prospective registration.
