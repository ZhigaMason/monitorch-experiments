#!/bin/bash
#PBS -N opt_neurons
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=32gb
#PBS -o logs/
#PBS -e logs/

cd $PBS_O_WORKDIR

source .venv/bin/activate

nvidia-smi

echo "SCIPRT=$SCRIPT"
echo "CONFIG=$CONFIG"

python $SCRIPT --config-file $CONFIG
