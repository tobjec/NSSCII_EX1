#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <mpi.h>

namespace program_options {

struct Options {
  unsigned int mpi_mode;  
  std::string name;
  size_t N;
  size_t iters;
  double fix_west;
  double fix_east;  
  void print() const {
    std::printf("mpi_mode: %u\nD", mpi_mode);    
    std::printf("name: %s\n", name.c_str());
    std::printf("N: %zu\n", N);
    std::printf("iters: %zu\n", iters);
    std::printf("fix_west: %lf\n", fix_west);
    std::printf("fix_east: %lf\n", fix_east);    
  }
};

auto parse(int argc, char *argv[]) {
  if (argc != 7)
    throw std::runtime_error("unexpected number of arguments");
  Options opts;
  if (std::string(argv[1]) == std::string("1D"))
    opts.mpi_mode = 1;
  else if( std::string(argv[1]) == std::string("2D"))
    opts.mpi_mode = 2;
  else
   throw std::runtime_error("invalid parameter for mpi_mode (valid are '1D' and '2D')");
  opts.name = argv[2];
  if (std::sscanf(argv[3], "%zu", &opts.N) != 1 && opts.N >= 2)
    throw std::runtime_error("invalid parameter for N");
  if (std::sscanf(argv[4], "%zu", &opts.iters) != 1 && opts.iters != 0)
    throw std::runtime_error("invalid parameter for iters");
  if (std::sscanf(argv[5], "%lf", &opts.fix_west) != 1)
    throw std::runtime_error("invalid value for fix_west");
  if (std::sscanf(argv[6], "%lf", &opts.fix_east) != 1)
    throw std::runtime_error("invalid value for fix_east");  
  return opts;
}

} // namespace program_options


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    // get global rank
    int grk;
    MPI_Comm_rank(MPI_COMM_WORLD, &grk);

    // get number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // special behaviour for one process
    bool flag_one_process = (size == 1);

    // one dimension
    constexpr int n = 1;

    // proper select of the dimensions
    std::array<int, n> dims = {0};
    MPI_Dims_create(size, n, std::data(dims));

    // create a communicator for 1-d cartesian topology
    std::array<int, n> periodic = {false};
    int reorder = true;
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), reorder, &comm);

    // get rank for comm
    int rk;
    MPI_Comm_rank(comm, &rk);

    // obtain neighbor ranks
    constexpr int displ = 1;
    std::array<int, n* 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL};
    enum Direction : int { Top = 0, Bottom = 1 };
    MPI_Cart_shift(comm, 0, displ, &nb[Bottom], &nb[Top]);

    // obtain own coordinates
    std::array<int, n> coord = {-1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    // parse args
    auto opts = program_options::parse(argc, argv);
    // opts.print();

    // define new MPI_Datatypes
    MPI_Datatype row;
    MPI_Type_vector(opts.N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);

    // calculating height data segment for given rank
    int height = -1;
    if (rk == 0) height = (int)(opts.N/size) + (opts.N%size) + 1;
    else if (rk == size-1) height = (int)(opts.N/size) + 1;
    else height = (int)(opts.N/size) + 2;
    /* divide the segment evenly among all processors; spill over is assigned to rank 0; +2/+1 because of two/one ghost layer*/
    // special behaviour in case of one process
    if (flag_one_process) height = opts.N;


    // initial guess (0.0) with fixed values in west and east
    auto init = [N = opts.N, W = opts.fix_west, E = opts.fix_east, height]() -> auto {
        std::vector<double> res(N * height);
        for (size_t j = 0; j < height; ++j)
        for (size_t i = 0; i < N; ++i) {
            res[i + j * N] = 0.0;
            if (i % N == 0)
            res[i + j * N] = W;
            if (i % N == N - 1)
            res[i + j * N] = E;
        }
        return res;
    };

    // solver update for only one process
    auto jacobi_iter_one_process = [N = opts.N](const auto &xold, auto &xnew,
                                    bool residual = false) {
        auto h = 1.0 / (N - 1);
        auto h2 = h * h;
        // all interior points
        for (size_t j = 1; j < N - 1; ++j) {
        for (size_t i = 1; i < N - 1; ++i) {
            auto w = xold[(i - 1) + (j)*N];
            auto e = xold[(i + 1) + (j)*N];
            auto n = xold[(i) + (j + 1) * N];
            auto s = xold[(i) + (j - 1) * N];
            auto c = xold[(i) + (j)*N];
            if (!residual)
            xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
            else
            xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
        }
        }
        // isolating south boundary
        {
        size_t j = 0;
        for (size_t i = 1; i < N - 1; ++i) {
            auto w = xold[(i - 1) + (j)*N];
            auto e = xold[(i + 1) + (j)*N];
            auto n = xold[(i) + (j + 1) * N];
            auto s = n;
            auto c = xold[(i) + (j)*N];
            if (!residual)
            xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
            else
            xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
        }
        }
        // isolating north boundary
        {
        size_t j = N - 1;
        for (size_t i = 1; i < N - 1; ++i) {
            auto w = xold[(i - 1) + (j)*N];
            auto e = xold[(i + 1) + (j)*N];
            auto s = xold[(i) + (j - 1) * N];
            auto n = s;
            auto c = xold[(i) + (j)*N];
            if (!residual)
            xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
            else
            xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
        }
        }
    };

    // solver update
    auto jacobi_iter = [N = opts.N, height, rk, size](const auto &xold, auto &xnew,
                                    bool residual = false) {
        auto h = 1.0 / (N - 1);
        auto h2 = h * h;
        if (rk == 0) {
            /* j=0: Dirichlet-Boundary; j=height-1: Ghost-Layer */
            // south boundary
            size_t j = 0;
            for (size_t i = 1; i < N - 1; ++i) {
                auto w = xold[(i - 1) + (j)*N];
                auto e = xold[(i + 1) + (j)*N];
                auto n = xold[(i) + (j + 1) * N];
                auto s = n;
                auto c = xold[(i) + (j)*N];
                if (!residual)
                xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                else
                xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
            }
            // all interior points
            for (size_t j = 1; j < (height-1); ++j) {
                for (size_t i = 1; i < N - 1; ++i) {
                    auto w = xold[(i - 1) + (j)*N];
                    auto e = xold[(i + 1) + (j)*N];
                    auto n = xold[(i) + (j + 1) * N];
                    auto s = xold[(i) + (j - 1) * N];
                    auto c = xold[(i) + (j)*N];
                    if (!residual)
                    xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                    else
                    xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
                }
            }
        }
        else if (rk == size-1) {
            /* j=0: Ghost-Layer; j=height-1: Dirichlet-Boundary */
            // North boundary
            size_t j = height - 1;
            for (size_t i = 1; i < N - 1; ++i) {
                auto w = xold[(i - 1) + (j)*N];
                auto e = xold[(i + 1) + (j)*N];
                auto s = xold[(i) + (j - 1) * N];
                auto n = s;
                auto c = xold[(i) + (j)*N];
                if (!residual)
                xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                else
                xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
            }
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < N - 1; ++i) {
                    auto w = xold[(i - 1) + (j)*N];
                    auto e = xold[(i + 1) + (j)*N];
                    auto n = xold[(i) + (j + 1) * N];
                    auto s = xold[(i) + (j - 1) * N];
                    auto c = xold[(i) + (j)*N];
                    if (!residual)
                    xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                    else
                    xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
                }
                }
        }
        else {
            /* j=0 and j=height-1: Ghost-Layer */
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < N - 1; ++i) {
                    auto w = xold[(i - 1) + (j)*N];
                    auto e = xold[(i + 1) + (j)*N];
                    auto n = xold[(i) + (j + 1) * N];
                    auto s = xold[(i) + (j - 1) * N];
                    auto c = xold[(i) + (j)*N];
                    if (!residual)
                        xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                    else
                        xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
                }
            }
        }
    };


    // write vector to csv
    auto write = [N = opts.N, name = opts.name](const auto &x) -> auto {
      std::ofstream csv;
      csv.open(name + ".csv");
      for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N - 1; ++i) {
          csv << x[i + j * N] << " ";
        }
        csv << x[(N - 1) + j * N];
        csv << "\n";
      }
      csv.close();
    };

    // 2 norm
    auto norm2 = [N = opts.N](const auto &vec) -> auto {
      double sum = 0.0;
      for (size_t j = 0; j < N; ++j)
        for (size_t i = 1; i < (N - 1); ++i)
          sum += vec[i + j * N] * vec[i + j * N];

      return std::sqrt(sum);
    };

    // Inf norm
    auto normInf = [N = opts.N](const auto &vec) -> auto {
      double max = 0.0;
      for (size_t j = 0; j < N; ++j)
        for (size_t i = 1; i < (N - 1); ++i)
          max = std::fabs(vec[i + j * N]) > max ? std::fabs(vec[i + j * N]) : max;
      return max;
    };

    // initialize x1 and x2
    auto x1 = init();
    auto x2 = x1;

    // perform jacobi iterations
    enum Request : int { Bottom_SEND = 0, Top_SEND = 1, Bottom_RECV = 2, Top_RECV = 3 };
    enum Tag : int { Bottom_Msg = 0, Top_Msg = 1 };
    for (size_t iter = 0; iter <= opts.iters; ++iter) {
      if (flag_one_process) jacobi_iter_one_process(x1, x2);
      else jacobi_iter(x1, x2);
      std::swap(x1, x2);

      // send ghost layer
      std::array<MPI_Request, n * 2 * 2> req;
      MPI_Isend(&x1[(1)*opts.N], 1, row, nb[Bottom], Bottom_Msg, comm, &req[Bottom_SEND]);
      MPI_Isend(&x1[(height-2)*opts.N], 1, row, nb[Top], Top_Msg, comm, &req[Top_SEND]);
      MPI_Irecv(&x1[0], 1, row, nb[Bottom], Top_Msg, comm, &req[Bottom_RECV]);
      MPI_Irecv(&x1[(height-1)*opts.N], 1, row, nb[Top], Bottom_Msg, comm, &req[Top_RECV]);

      // wait/block for/until all four communication calls to finish
      std::array<MPI_Status, n * 2 * 2> status;
      MPI_Waitall(n * 2 * 2, std::data(req), std::data(status));
    }

    // calculating residual in x2
    if (flag_one_process) jacobi_iter_one_process(x1, x2, true);
    else jacobi_iter(x1, x2, true);

    // use MPI_Gather to collect all partial results
    auto gather_all_parts = [N = opts.N, rk, size, height, comm, row](const auto& vec) -> auto {
      // initialising receive buffer
      std::vector<double> recvbuf;
      if (rk == 0) {
        recvbuf = vec;
        recvbuf.resize(N*N, -1);
      }
      // gather different parts of different processes into process rk=0
      if (rk == 0)
        MPI_Gather(&vec[(N%size)*N], height-1-(N%size), row, std::data(recvbuf)+((N%size)*N), height-1-(N%size), row, 0, comm);
      else if (rk == size-1)
        MPI_Gather(&vec[(1)*N], height-1, row, std::data(recvbuf)+((N%size)*N), height-1, row, 0, comm);
      else
        MPI_Gather(&vec[(1)*N], height-2, row, std::data(recvbuf)+((N%size)*N), height-2, row, 0, comm);

      return recvbuf;
    };

    std::vector<double> solution;
    std::vector<double> residual;

    if (flag_one_process) {
      solution = x1;
      residual = x2;
    }
    else {
      // perform gather for solution and residual
      solution = gather_all_parts(x1);
      residual = gather_all_parts(x2);
    }
    
    if (rk == 0) {
      write(solution);
      std::cout << "  norm2 = " << norm2(residual) << std::endl;
      std::cout << "normInf = " << normInf(residual) << std::endl;
    }

    MPI_Comm_free(&comm);
    MPI_Finalize();

    return 0;
}