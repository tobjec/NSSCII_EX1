jacobi.o: main.cpp
	mpic++ -std=c++17 -O3 -march=native jacobi.cpp -o jacobi.o

.PHONY: load clean run plot
load:
	source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64

run:
	mpirun -n $(filter-out $@,$(MAKECMDGOALS)) ./jacobi.o result 80 10000 -100 100

plot:
	python3 plot.py

clean:
	rm -f jacobi.o
	rm -f result*.csv
