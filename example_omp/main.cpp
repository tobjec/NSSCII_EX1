#include <iostream>
#include <omp.h> // https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf

int main() {

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
#pragma omp critical
    {
      std::cout << "printing to stdout from tid=" << tid << "." << std::endl;
      std::cerr << "printing to stderr from tid=" << tid << "." << std::endl;
    }
  }

  return 0;
}
