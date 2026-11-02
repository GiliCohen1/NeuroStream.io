# Held-out EEG test recordings

75 single-channel EEG recordings (4097 samples each, 173.61 Hz) from the University of Bonn
dataset. They are the recordings the model **never saw during training** (see
`models/heldout_test_files.json`), included only so the demo and CI end-to-end check can replay
them. Set `s` = recorded during seizures; sets `z`, `o`, `n`, `f` = non-seizure.

Source: Andrzejak RG, Lehnertz K, Mormann F, Rieke C, David P, Elger CE (2001).
"Indications of nonlinear deterministic and finite-dimensional structures in time series of brain
electrical activity: Dependence on recording region and brain state." Phys Rev E 64, 061907.
Original data: https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/
Per the source page the data is free for research and educational use. If you reuse these files,
cite the paper above. The full dataset can be fetched with `python training/download_data.py`.
