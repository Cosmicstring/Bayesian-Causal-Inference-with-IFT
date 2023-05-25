# Bayesian Causal Inference project using NIFTy

This repo contains all the code implemented and described in this [paper](https://gitlab.mpcdf.mpg.de/ift/papers/bayesiancausalinference). In order to run the code one needs all the standard `python3` libraries and the [NIFTy6](https://gitlab.mpcdf.mpg.de/ift/nifty).

All the models are implemented in the `bipartite_model.py` and `confounder_model.py` with their shared options implemented inside `causal_model.py`, which also contains the method for estimating model evidences (the `_get_evidence(**args)`). For changing the hyperparameters of the models themselves take a look into the `config.json` file. The `data_processing_utilities.py` contains the relevant setup information for configuring the code arguments (look into `Parser`). Main pyscript is `do_bci_inference.py` which does the actual tests.

An example of how to run the tests and select models / testcases is presented inside the `evaluation_scripts/run.sh` bash script. First copy it to the $ROOT folder containing the `do_bci_inference.py` script and run it. For any problems contact me at `akostic@mpa-garching.mpg.de`. 
