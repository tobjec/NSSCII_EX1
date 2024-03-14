#!/bin/bash

## job name

#SBATCH --job-name=single_thread

## logfiles (stdout/stderr) %x=job-name %j=job-id

#SBATCH --output=stdout-%x.%j.log
#SBATCH --error=stderr-%x.%j.log

## resource requests 

#SBATCH --partition=nssc    # partition for 360.242 and 360.242
#SBATCH --nodes=1           # request one node
#SBATCH --ntasks=1          # request one process on this node
#SBATCH --cpus-per-task=1   # request one cpu for this process
#SBATCH --time=00:00:20     # set time limit to 20 seconds

## load modules and compilation (still on the login node)

g++ -std=c++17 main.cpp -o main

## submitting jobs (on the allocated resources)

# job: run the compiled executable

./main 
