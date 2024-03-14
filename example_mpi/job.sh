#!/bin/bash

## job name

#SBATCH --job-name=mpi

## logfiles (stdout/stderr) %x=job-name %j=job-id

#SBATCH --output=stdout-%x.%j.log
#SBATCH --error=stderr-%x.%j.log

## resource requests 

#SBATCH --partition=nssc    # partition for 360.242 and 360.242
#SBATCH --nodes=1           # request one node
#SBATCH --ntasks=40         # request 40 processes on this node
#SBATCH --cpus-per-task=1   # request one cpu for each of these processes
#SBATCH --time=00:00:20     # set time limit to 20 seconds

## load modules and compilation (still on the login node)

module load mpi/openmpi-x86_64  # MPI implementation (including compiler-wrappers mpicc/mpic++)

mpic++ -std=c++17 main.cpp -o main

## submitting jobs (on the allocated resources)

# job: run the mpi-enabled executable

mpirun ./main # will issue --ntasks=40 processes
