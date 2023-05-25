#!/usr/bin/env sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

rm save_pid.txt
benchmark='tcep_no_des'
for model in "X->Y" "Y->X"
do
    for i in {0..10}
    do
        rm run_$model\_$i\_${benchmark}.out
        nohup python3 do_bci_inference.py --config 'config.json' --benchmark ${benchmark} --direction $model --batch $i --N_samples 2 --N_steps 30 --analyse 0 --version 'v2' >> run_$model\_$i\_${benchmark}.out &
        echo $! >> save_pid.txt
    done
done
