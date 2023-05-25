import os
import numpy as np

ROOT = "/freya/ptmp/mpa/akostic/andrija-kostic/Bayesian_Causal_Inference/implementation/"

git = 60 # number of global iterations
starts = 2 # start number of mirrored samples
benchmark = 'tcep_no_des' # benchmark tests to be used
benchmark_size = 10

if 'tcep' in benchmark:
    benchmark_size += 1 

model_version = 'v3'
analyse = 0
config_file = 'config.json'

#shared_options = "--time=23:50:00 "
#shared_options += "--mem=10G "
#shared_options += "--mail-user=akostic@mpa-garching.mpg.de "
#shared_options += "--mail-type=NONE "
#shared_options += "-p p.24h "
#shared_options += "-D ./ "
#shared_options += ROOT + "run_on_freya.sh "
path_out = ROOT + 'freya_output/' + benchmark + '/'

if not os.path.exists(path_out):
    os.mkdir(path_out)

shared_options = "#!/bin/bash -l \n#\n"
shared_options += "# Standard output and error: \n"
shared_options += "#SBATCH -o ./tjob.out.%j \n"
shared_options += "#SBATCH -e ./tjob.err.%j \n"
shared_options += "# Initial working directory: \n"
shared_options += "#SBATCH -D ./ \n"
shared_options += "# Job Name: \n "
shared_options += "#SBATCH -J bci_m_benchmark_{}_version_{} \n ".format(benchmark, model_version)
shared_options += "# Queue (Partition): \n "
shared_options += "#SBATCH --partition=p.24h \n "
shared_options += "# Number of nodes and tasks per node: \n "
shared_options += "#SBATCH --nodes=1 \n "
shared_options += "#SBATCH --ntasks={} \n ".format(benchmark_size)
shared_options += "#SBATCH --cpus-per-task=1 \n "
shared_options += "#SBATCH --mail-type=END \n "
shared_options += "#SBATCH --mail-user=akostic@mpa-garching.mpg.de \n "
shared_options += "#SBATCH --time=23:50:00 \n\n\n"

shared_options += "export OMP_NUM_THREADS=1 \n"
shared_options += "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} \n"
shared_options += "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK} \n"
shared_options += "export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK} \n"
shared_options += "export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK} \n"
shared_options += "export OMP_PLACES=cores \n\n\n"

def make_job(model, batch, model_comment):
    output = "{}_minimizingjob.err ".format(path_out + model_comment + "_" + model_version + "_" + str(batch))
    cmd = "(cd {} ; \ \n".format(ROOT)
    cmd += "./run_on_freya.sh "
    cmd += "--config '{:s}' ".format(config_file)
    cmd += "--benchmark '{:s}' ".format(benchmark)
    cmd += "--direction '{:s}' ".format(model)
    cmd += "--batch {:d} ".format(batch)
    cmd += "--N_samples {:d} ".format(starts)
    cmd += "--N_steps {:d} ".format(git)
    cmd += "--analyse {:d} ".format(analyse)
    cmd += "--version '{:s}' ".format(model_version) 
    cmd += "&> {} ) &\n\n\n".format(output)
    
    print(cmd)
    command.write(cmd)

if __name__=="__main__":
    models = ["X->Y", "Y->X"]
    model_comments = ["X-\>Y", "Y-\>X"]
    submit_comments = ["XY", "YX"]
    
    if 'tcep' in benchmark:
        batches = np.arange(11)
    elif ('SIM' in benchmark) or ('bcs' in benchmark):
        batches = np.arange(1,11)

    # do the confounder
    #for model, model_comment in zip([models[0]], [model_comments[0]]):
    #    for batch in batches:
    #        submit_minimizing_job(model, batch, model_comment)
    
    for model, model_comment, submit_comment in zip(models, model_comments, submit_comments):
        submit = "submit_{}_{}_{}_cont.sh".format(submit_comment, model_version, benchmark)
        command = open(submit, "w")
        command.write(shared_options)    
    
        for batch in [batches[7]]:
            make_job(model, batch, model_comment)
            
        command.write("wait\n")
        command.close()

        #os.popen("sbatch " + submit)
