# What is reproduced

This release is tied to paper version 1, DOI 10.5281/zenodo.23284178.

1. Task-feature reproduction: original numerical functions are applied to minimally scoped task-level inputs and EEG feature arrays. The all-participant endpoints, donor comparisons, phase axes, and band-group endpoints are compared to retained original tables.
2. Null recalibration: participant, band-group, lightlike-source, and post-midpoint p-values are recomputed from retained per-null rows. Unless explicitly listed otherwise, these null draws are not generated anew by the default commands.
3. Null regeneration: the core EEG command regenerates all 2,000 P1 order permutations using the original seed and compares every row. It recalculates all 26 cyclic shifts for all 26 participants. Full mode recalculates responses to all 1,000 retained random-kernel ensembles; it does not resample a different ensemble.
4. Finite source: default mode rebuilds the ideal observer family and responses from retained compressed states. Full mode reconstructs the finite matrices, spectral transfer, original Gram, and compressed states first. Rational response matrices, fingerprint matrices, traces, and ranks are compared exactly.
5. Record compiler: the original exact-rational polynomial interval certificate is rerun against the identical frozen source bytes. All eight ideal complementary purities remain distinct; the 24 directed updates and 24 two-order histories are checked against the fixed cube.

Numerical tolerances are 1e-12 for the core endpoints. Phase-source comparisons use the original frozen phase driver's `BASELINE_TOL=1e-10`. Exact finite-source comparisons use equality over rational/radical pairs, not a floating tolerance. New outputs are written separately from retained references.

# Boundaries

The repository does not experimentally test the ontological claims, the subjective content of experience, actual ion-channel coherence, or coupling from a quantum source into a human body. The normalized transfer, reset-model, and general recovery conditions in §§5.10–5.11 and appendices are analytic arguments, not a separate empirical dataset reproduced here. The six-configuration Kv1.2-inspired coefficient example is a producer calculation with independent audit still pending in the paper; the release does not upgrade it.

Historical independent audits remain historical records. Preparing and executing this package is a producer reproduction check, not a new independent E2 audit or independent-data replication.

The raw-waveform acquisition/preprocessing pipeline belongs to the cited prior data lineage. Released features suffice to reproduce the listed paper-specific endpoints. The original raw EEG is not duplicated here.
