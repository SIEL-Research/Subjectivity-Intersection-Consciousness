# Released numerical inputs

- `task_features.csv`: 676 rows, 26 pseudonymous participants × 26 ordered analysis units. `p00,p01,p10,p11` are four quantum outcome probabilities; `I2` gives the squared quantum radius. `lambda2_4ch,lambda3_4ch,lambda4_4ch` sum to Lambda. `h1_total_persistence,aspl,lambda2` define the 14-electrode topology.
- `source_kernel_records.json`: eight fixed source masks in order 0,5,8,13,16,21,24,29; fixed normalized Gram/correlation kernels.
- `signed_source_features.csv`, `full_receiver_features.csv`: original derived numerical source/receiver inputs for state and phase comparisons. `target_task` indexes an analysis unit.
- `quantum_band_features.csv`: numerical quantum and EEG band summary columns used by the original light/time/space source-response functions. Unused nonnumeric columns are omitted.
- `eeg_band_feature_arrays.npz`: participant order P1…P26, unit order 0…25. `Heart` contains seven left/right asymmetry coordinates, `Mind` contains the earlier two state features, `Body` contains 91 spatial-separation coordinates, `bands` contains Delta,Theta,Alpha,Beta,Gamma means, and `energies` contains the fixed nuisance energy. The earlier Mind features belong to the original three-group comparison; the revised temporal-direction Beta result is in the next file.
- `ordered_update_arrays.npz`: `update` and Delta/Theta/Alpha/Beta/Gamma outcomes, each 26×26, retaining the original energy-residualized inputs to the forward-minus-reverse analysis.

Reference CSV and compressed CSV files retain all reported comparator/null rows, not only significant rows. `reference/finite_source/` retains exact rational/radical-pair state and response records. A pair `[a,b]` denotes `a+b*sqrt(17)`.

Participant labels are pseudonymous. No mapping to real names or author identity is supplied.
