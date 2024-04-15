#!/bin/bash

## job name

#SBATCH --job-name=cpu24

## logfiles (stdout/stderr) %x=job-name %j=job-id

#SBATCH --output=stdout-%x.%j.log
#SBATCH --error=stderr-%x.%j.log

## resource requests 

#SBATCH --partition=nssc    # partition for 360.242 and 360.242
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks=24          # number of tasks
#SBATCH --cpus-per-task=1  # number of cpus per process
#SBATCH --time=00:10:00     # set time limit to 1 minute

## load modules and compilation (still on the login node)

module load pmi/pmix-x86_64     # [P]rocess [M]anagement [I]nterface (required by MPI-Implementation)
module load mpi/openmpi-x86_64  # MPI implementation (including compiler-wrappers mpicc/mpic++)

mpic++ -std=c++17 -O3 -pedantic -march=native -ffast-math impl_1d.cpp -o impl_1d

## submitting jobs (on the allocated resources)

# job: run the mpi-enabled executable passing command line arguments

mpi_mode=1D
filename=cpu24
resolutions=(125 250 1000 2000)
iterations=800

for resolution in "${resolutions[@]}"
do
  mpirun ./impl_1d ${mpi_mode} ${filename}_res${resolution} ${resolution} ${iterations} -100 +100
done

