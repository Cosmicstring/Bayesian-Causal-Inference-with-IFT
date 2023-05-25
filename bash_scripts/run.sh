#!/usr/bin/env sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 do_bci_inference.py --config 'config.json' --benchmark 'synthetic' --direction "Y->X" --batch 0 --N_samples 2 --N_steps 2 --analyse 0 --version 'v2'
