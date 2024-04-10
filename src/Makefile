jacobi.o: main.cpp
	mpic++ -std=c++17 -O3 -march=native jacobi.cpp -o jacobi.o

.PHONY: load clean run plot
load:
	source /etc/profile.d/modules.sh && module load mpi/openmpi-x86_64

run:
	mpirun -n $(filter-out $@,$(MAKECMDGOALS)) ./jacobi.o result 80 10000 -100 100
# $(MAKECMDGOALS) is the list of "targets" spelled out on the command line, e.g. "run value1 value2".
# $@ is an automatic variable for the name of the target of the rule, in this case "run".
# filter-out is a function that removes some elements from a list. So $(filter-out bar, foo bar baz) returns foo baz (it can be more subtle, but we don't need subtlety here).
# Put these together and $(filter-out $@,$(MAKECMDGOALS)) returns the list of targets specified on the command line other than "run", which might be "value1 value2".
# see also https://stackoverflow.com/questions/6273608/how-to-pass-argument-to-makefile-from-command-line

plot:
	python3 plot.py

clean:
	rm -f jacobi.o
	rm -f result*.csv


