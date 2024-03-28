# NSSCII_EX1
First group homework of the lecture 360.243 Numerical Simulation and Scientific Computing II (VU 3,0) 2024S

Link to LaTeX template for task reports:
https://www.overleaf.com/6217539947gjpfdrqgnncz#a5cfc3

Lecture notes:

* first do 1D decomposition and only after working example implement 2D decomposition
* make use of the proper MPI functions for 2D to make code 


## Task 1

1. 
2. After one iteration depending on the decomposition of the domain the results will not be numerically identical. E. g. with this practice if you divide the domain vertically the inner regions will have 0s as east and west ghost boundaries thus all values will be 0. Only after a few iterations the results from the outer most boudaries (e.g. east west fixed values) will be passed onto the inner domains. 
3. If you are on a big distributed system with limited network capacity it may prove advantageous to do several iterations before needing to communicate again.
4. 1MB/core => 20MB/CPU => 40MB/System => 80MB for 2 Systems
