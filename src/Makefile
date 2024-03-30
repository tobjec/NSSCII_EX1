jacobi.o: main.cpp
	mpic++ -std=c++17 -O3 -march=native jacobi.cpp -o jacobi.o

.PHONY: clean run
run:
	mpirun -n 8 ./jacobi.o result 50 10000 -100 100
	
clean:
	rm -f jacobi.o
	rm -f result.csv
