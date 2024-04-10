#!/bin/bash
#SBATCH --job-name=1_tv_synth_p2p          # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail
#SBATCH --mail-user=nc1514@princeton.edu

module purge
module load anaconda3/2023.9
conda activate torch-env

python train.py --dataroot ../outputs/hdf5/img_2_img_proof.h5 --name tv_inv_proof --model pix2pix --direction BtoA --dataset_mode hdf5 --gpu_ids 0,1,2,3 --batch_size 4 --norm instance --display_id 0 --input_nc 1 --output_nc 1

# ,1,2,3 --batch_size 4 --norm instance
