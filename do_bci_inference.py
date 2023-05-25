# References:
#
# [1] Kurthen, Maximilian, and Torsten A. Enßlin. "Bayesian Causal Inference." arXiv preprint arXiv:1812.09895 (2018).
#
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import nifty6 as ift
import data_processing_utilities as dpu
from data_processing_utilities import Parser, neglected_files

from causal_model import Causal_Model
from select_model import select_model

parser = Parser()
args = parser.parse_args()

N_samples = args.N_samples
N_steps = args.N_steps
analyse = args.analyse
config = args.config
version = args.version
benchmark = args.benchmark
direction = args.direction
batch = args.batch

file_setup = open(config, "r")
setup = json.load(file_setup)
file_setup.close()

TEST_ROOT = 'benchmark_tests/' + benchmark
test_file_list = \
    os.listdir(TEST_ROOT)

test_file_list.sort()

causality_output_root = benchmark + '_tests/'

if not os.path.exists(causality_output_root):
    os.mkdir(causality_output_root)

causality_output = \
    causality_output_root + 'N_bins{}'.format(setup['real_model']['Nbins'])

if not os.path.exists(causality_output):
    os.mkdir(causality_output)

f = open(causality_output +
         '/Evidence_benchmark_{}_direction_{}_version_{}_batch_{}_steps_{}.txt'.format(\
                                                            benchmark,
                                                            direction,
                                                            version,
                                                            batch,\
                                                            N_steps), 'w')

for test_filename in test_file_list[parser.batches(args)]: 

    # Exclude the files which have multiple dimensional data
    if not (test_filename in neglected_files[benchmark]):
        print(test_filename)
        ln_evidence = {}

        X, Y = dpu.get_data(setup, test_filename, TEST_ROOT)

        # Setup the directory for test results
        test_output = benchmark + '_tests/{}'
        test_output = test_output.format(test_filename[:-4])

        if not os.path.exists(test_output):
            os.mkdir(test_output)
            global_output_path = test_output + '/{}'
        else:
            global_output_path = test_output + '/{}'

        current_output_path = global_output_path.format(
            "N_samples_{}_N_steps_{}")
        current_output_path = current_output_path.format(
            N_samples, N_steps)

        if not os.path.exists(current_output_path):
            os.mkdir(current_output_path)
        
        if not os.path.exists(current_output_path + "/samples/"):
            os.mkdir(current_output_path + "/samples/")

        current_output_path = current_output_path + '/{}'

        cm = Causal_Model(direction, data=[X, Y], config=setup, version=version)
        model = select_model(cm)
        
        filename = "BCI_{}_version_{}_{}.pdf".format(direction, model.version, '{}')
        
        if analyse == 1:

            import glob
            from model_utilities import load_KL_sample, load_KL_position

            path = benchmark + "_tests/" + test_filename[:-4]
            path += "/N_samples_{}_N_steps_{}/".format(N_samples, N_steps) 

            seed = model.model_dict['seed']
            f_ID = path + direction + "_" + "KL_position_version_{}_{}.npy".format(version, seed) 
            KL_position = load_KL_position(f_ID)

            KL_position = ift.makeField(model._Ham.domain, KL_position)

            f_ID = path + "samples/" + direction + "*version_{}*_{}_*".format(version, seed)
            samples = glob.glob(f_ID)

            positions = []
            for file in samples:
                sample = load_KL_sample(file)
                sample = ift.makeField(\
                            model._Ham.domain, sample)    
                positions.append(KL_position + sample)

            output_path = "analyse_results/" + direction

            output_path = output_path + "_{}.pdf".format(version)

            # model._plot_setup(\
            #    output_path, positions)

            KL = ift.MetricGaussianKL(KL_position, model._Ham, n_samples=1)
            
            # FIXME: Here the rstate should be reloaded to be consistent with the
            # previous evidence caluclation (?) Or maybe leave it without rstate
            # save because the only priority is to resample the posterior somehow
           
            try:
                ln_evidence[direction] = model._get_evidence(KL,
                        n_eigs = model._k_indx(positions)) 
            except:
                Warning("Testcase {} on {} failed".format(test_filename, direction))
                ln_evidence[direction] = model.fail_dictionary
                pass
            
            for causal_dir, val in ln_evidence.items():
                print(\
                        ("{:s}: \n mean : {:.5e} , upper : {:.5e} , lower : "+
                         "{:.5e}\n H_lh : {:.5e} +- {:.5e} \n xi2 : {:.5e} \n"+
                         " Tr_reduce_Lambda : {:.5e} (+{:.5e})\n Tr_Ln_Lambda : "+
                        "{:.5e} (+{:.5e})\n").format(
                        causal_dir, \
                        val['mean'], val['upper'], val['lower'],\
                        val["H_lh"], val["var_H_lh"], val["xi2"],\
                        val["Tr_reduce_Lambda"],val["err_TrL"],val["Tr_ln_Lambda"],\
                        val["err_TrlnL"]))
        else:

            model.plot_initial_setup(current_output_path.format(filename))
            
            try:
                ln_evidence[direction] = \
                        model.optimize_and_get_evidence(\
                            N_samples, N_steps,
                            track_optimization=True,
                            current_output_path=current_output_path,
                            filename=filename,
                            point_estimates=model.point_estimates,
                            plot_final=True)[0]
            except:
                Warning("Testcase {} on {} failed".format(test_filename, direction))
                ln_evidence[direction] = model.fail_dictionary
                pass

        f.write("---------------------------------\n")
        f.write("file:\n")
        f.write("{:s}\n".format(test_filename))

        # Write the evidence values alongside with their bounds
        # marking the 'max_model_key' chosen above with '**'
        for causal_dir, val in ln_evidence.items():
            f.write(\
                    ("{:s}: \n mean : {:.5e} , upper : {:.5e} , lower : "+
                     "{:.5e}\n H_lh : {:.5e} +- {:.5e} \n xi2 : {:.5e} \n"+
                     " Tr_reduce_Lambda : {:.5e} (+{:.5e})\n Tr_Ln_Lambda : "+
                    "{:.5e} (+{:.5e})\n").format(
                    causal_dir, \
                    val['mean'], val['upper'], val['lower'],\
                    val["H_lh"], val["var_H_lh"], val["xi2"],\
                    val["Tr_reduce_Lambda"],val["err_TrL"],val["Tr_ln_Lambda"],\
                    val["err_TrlnL"]))                

        f.write("---------------------------------\n")
# Output the evidence calculation into a file:
f.close()
