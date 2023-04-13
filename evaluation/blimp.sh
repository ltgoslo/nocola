#!/bin/bash

#SBATCH --job-name=NoCoLA
#SBATCH --account=ec30
#SBATCH --time=04:00:00  # Max walltime is 150 hours.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  # 6 CPU cores per task to keep the parallel data feeding going. A little overkill, but CPU time is very cheap compared to GPU time.
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# Load modules
module --quiet purge  # Reset the modules to the system default
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source /cluster/work/projects/ec30/davisamu/pytorch_1.11.0/bin/activate

python3 blimp.py
