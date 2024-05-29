#!/bin/bash
#SBATCH --job-name=example
#SBATCH --output=./example.out
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=64
mpiexec -np 64 python ./optimal_estimation_example.py && exit