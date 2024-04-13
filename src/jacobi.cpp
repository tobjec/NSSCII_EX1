// g++ solver.cpp -std=c++17 -O3 -march=native -ffast-math -o solver
// ./solver without_source 10 10000 0.0 1.0
// ./solver with_source 50 100000 0.0 0.0 0.5 0.5 0.1

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>
#include <algorithm>
#include <array>
#include <sstream>

#include <mpi.h>

//using namespace std::chrono;

namespace program_options {

struct Options {
  std::string name;
  size_t N;
  size_t iters;
  double fix_west;
  double fix_east;
  bool has_source;
  double source_x;
  double source_y;
  double source_sigma;
  void print() const {
    std::printf("name: %s\n", name.c_str());
    std::printf("N: %zu\n", N);
    std::printf("iters: %zu\n", iters);
    std::printf("fix_west: %lf\n", fix_west);
    std::printf("fix_east: %lf\n", fix_east);
    std::printf("has_source: %s\n", has_source ? "true" : "false");
    std::printf("source_x: %lf\n", source_x);
    std::printf("source_y: %lf\n", source_y);
    std::printf("source_sigma: %lf\n", source_sigma);
  }
};

auto parse(int argc, char *argv[]) {
  if (argc != 9 && argc != 6)
    throw std::runtime_error("unexpected number of arguments");
  Options opts;
  opts.name = argv[1];
  if (std::sscanf(argv[2], "%zu", &opts.N) != 1 && opts.N >= 2)
    throw std::runtime_error("invalid parameter for N");
  if (std::sscanf(argv[3], "%zu", &opts.iters) != 1 && opts.iters != 0)
    throw std::runtime_error("invalid parameter for iters");
  if (std::sscanf(argv[4], "%lf", &opts.fix_west) != 1)
    throw std::runtime_error("invalid value for fix_west");
  if (std::sscanf(argv[5], "%lf", &opts.fix_east) != 1)
    throw std::runtime_error("invalid value for fix_east");
  if (argc == 6) {
    opts.has_source = false;
    opts.source_x = NAN;
    opts.source_y = NAN;
    opts.source_sigma = NAN;
    return opts;
  }
  if (std::sscanf(argv[6], "%lf", &opts.source_x) != 1)
    throw std::runtime_error("invalid value for source_x");
  if (std::sscanf(argv[7], "%lf", &opts.source_y) != 1)
    throw std::runtime_error("invalid value for source_y");
  if (std::sscanf(argv[8], "%lf", &opts.source_sigma) != 1)
    throw std::runtime_error("invalid value for source_sigma");
  opts.has_source = true;
  return opts;
}

} // namespace program_options
double norm_2(const std::vector<double>& v) {
  auto sum = 0.;
  for (auto x : v) {
    sum += x * x;
  }

  return std::sqrt(sum);
}

double norm_inf(const std::vector<double>& v) {
  auto max = 0.;
  for (auto x : v) {
    max = std::abs(x) > max ? std::abs(x) : max;
  }

  return max;
}

std::vector<double> sub(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> result(a.size());
  for (auto i = 0; i < result.size(); ++i) {
    result[i] = a[i] - b[i];
  }

  return result;
}

auto write(const std::vector<double>& u_h_out, size_t& N, size_t start, size_t end, std::string name) {
  std::ofstream csv;
  csv.open(name + ".csv");
  for (size_t i = start; i < end; ++i) {
    if (i != start && i % N == 0) csv << "\n";
    csv << std::format("{:08.4f}", u_h_out[i]);
    if (i % N != N-1) csv << " ";
  }
  csv << "\n";
  csv.close();
}

int main(int argc, char *argv[]) try {
  // first thing to do in an MPI program
  MPI_Init(&argc, &argv);
  // obtain own global rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // obtain number of global MPI processes
  int processes;
  MPI_Comm_size(MPI_COMM_WORLD, &processes);

  // letting MPI find a proper select of the dimensions (obsolete for one-dimension)
  constexpr int n = 1; // dimensions
  std::vector<int> dims(n);
  std::fill(dims.begin(), dims.end(), 0);
  MPI_Dims_create(processes, n, std::data(dims));

  // create a communicator for a Nth D topology (with potential reordering of ranks)
  MPI_Comm comm;
  std::vector<int> periodic(n);
  std::fill(periodic.begin(), periodic.end(), false);
  MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), true, &comm);

  // obtain and store neighboring ranks in the 2D topology (left/right)
  constexpr int displ = 1;
  enum Direction : int { LEFT = 0, RIGHT = 1, TOP = 2, BOTTOM = 3 };
  std::vector<int> neighbours(n*2);
  std::fill(neighbours.begin(), neighbours.end(), MPI_PROC_NULL);
  MPI_Cart_shift(comm, 0, displ, &neighbours[TOP], &neighbours[BOTTOM]);
  //MPI_Cart_shift(comm, 1, displ, &neighbours[LEFT], &neighbours[RIGHT]);
  
  // obtain own coordinates
  std::vector<int> coords(n);
  std::fill(coords.begin(), coords.end(), -1);
  MPI_Cart_coords(comm, rank, n, std::data(coords));

  // parse cmd options
  auto opts = program_options::parse(argc, argv);

  // start clock
  const auto start = std::chrono::high_resolution_clock::now();

  // HORIZONTAL DECOMPOSITION
  // lets ignore floats for now. thus grid size divided by number of MPI processes must be an integer
  size_t slice = opts.N / processes + 2; // +2 for ghost layers


  const auto grid_size = opts.N * slice;
  std::vector<double> u_h_in(grid_size), u_h_out(grid_size), b_h(grid_size), b_h_ref(grid_size);

  std::fill(b_h.begin(), b_h.end(), 0.);

  //BEGIN solve section
  // use zero-vector as initial solution
  std::fill(u_h_in.begin(), u_h_in.end(), 0.);
  //for (auto i=0;i<opts.N*opts.N;i+=opts.N) u_h_in[i] = opts.fix_west;
  //for (auto i=opts.N-1;i<opts.N*opts.N;i+=opts.N) u_h_in[i] = opts.fix_east;
  //for (auto i=0;i<opts.N*opts.N;i+=opts.N) u_h_out[i] = opts.fix_west;
  //for (auto i=opts.N-1;i<opts.N*opts.N;i+=opts.N) u_h_out[i] = opts.fix_east;

  // register MPI data types (here vectors) to send a row or a column
  MPI_Datatype row;
  MPI_Type_vector(opts.N, 1, 1, MPI_DOUBLE, &row);
  MPI_Type_commit(&row);
  MPI_Datatype col;
  MPI_Type_vector(opts.N, 1, opts.N, MPI_DOUBLE, &col);
  MPI_Type_commit(&col);
  
  // send own middle column to both neighbors
  // recieve a column from left/right neighbor and store as right/left column
  enum Request : int { TOP_SEND = 0, BOTTOM_SEND = 1, TOP_RECV = 2, BOTTOM_RECV = 3 };//, LEFT_SEND = 4, LEFT_RECV = 5, RIGHT_SEND = 6, RIGHT_RECV = 7 };
  enum Tag : int { TOP_tag = 0, BOTTOM_tag = 1 };//, LEFT_tag = 2, RIGHT_tag = 3 };
  std::vector<MPI_Request> req(n * 2 * 2);
  //std::fill(req.begin(), req.end(), MPI_REQUEST_NULL);

  // Jacobi iteration
  for (auto iter = 0; iter < opts.iters; ++iter) {
    //for (auto i = coords[1] * vertical_size; i < (coords[1] + 1) * vertical_size; ++i) {
    for (auto i = 0; i < opts.N; ++i) {
      for (auto j = 0; j < slice; ++j) { // first and last row are ghost layers
        const auto u_C =                  u_h_in[j       * opts.N + i      ];
        const auto u_W = i > 0          ? u_h_in[j       * opts.N + (i - 1)] : opts.fix_west;
        const auto u_E = i < opts.N - 1 ? u_h_in[j       * opts.N + (i + 1)] : opts.fix_east;
        const auto u_S = j > 0          ? u_h_in[(j - 1) * opts.N + i      ] : coords[0] == 0             ? u_C : u_h_in[(j - 1) * opts.N + i];
        const auto u_N = j < slice - 1  ? u_h_in[(j + 1) * opts.N + i      ] : coords[0] == processes - 1 ? u_C : u_h_in[(j + 1) * opts.N + i];
        u_h_out[j * opts.N + i] = (1. / (opts.N - 1.)) * (1. / (opts.N - 1.)) / 4. * (b_h[j * opts.N + i] + 1. / (1. / (opts.N - 1.)) / (1. / (opts.N - 1.)) * (u_W + u_E + u_S + u_N));
      }

      MPI_Isend(&u_h_out[opts.N], 1, row, neighbours[TOP], TOP_tag, comm, &req[TOP_SEND]);
      MPI_Isend(&u_h_out[opts.N*(slice-2)], 1, row, neighbours[BOTTOM], BOTTOM_tag, comm, &req[BOTTOM_SEND]);
      MPI_Irecv(&u_h_out[opts.N*(slice-1)], 1, row, neighbours[BOTTOM], TOP_tag,    comm, &req[TOP_RECV]);
      MPI_Irecv(&u_h_out[0],    1, row, neighbours[TOP],    BOTTOM_tag, comm, &req[BOTTOM_RECV]);
      
      // wait/block for/until all four communication calls to finish
      MPI_Waitall(n * 2 * 2, std::data(req), MPI_STATUSES_IGNORE);
    }

    // The next iteration's input is the current iteration's output
    std::swap(u_h_in, u_h_out);
  }
  //END solve section

  // Recompute
  // compute A_h * u_h = b_h with A_h given implicitly by the stencil and boundary conditions
  //for (auto i = coords[1] * vertical_size; i < (coords[1] + 1) * vertical_size; ++i) {
  for (auto i = 0; i < opts.N; ++i) {
    for (auto j = 0; j < slice; ++j) { // first and last row are ghost layers
      const auto u_C =                  u_h_out[j       * opts.N + i      ];
      const auto u_W = i > 0          ? u_h_out[j       * opts.N + (i - 1)] : opts.fix_west;
      const auto u_E = i < opts.N - 1 ? u_h_out[j       * opts.N + (i + 1)] : opts.fix_east;
      const auto u_S = j > 0          ? u_h_out[(j - 1) * opts.N + i      ] : coords[0] == 0             ? u_C : u_h_out[(j - 1) * opts.N + i];
      const auto u_N = j < slice - 1  ? u_h_out[(j + 1) * opts.N + i      ] : coords[0] == processes - 1 ? u_C : u_h_out[(j + 1) * opts.N + i];

      b_h_ref[j * opts.N + i] = -1. / (1. / (opts.N - 1.)) / (1. / (opts.N - 1.)) * (u_W + u_E + u_S + u_N - 4 * u_C);
    }
  }

  //const auto diff = sub(b_h, b_h_ref);

  // stop clock and calculate duration
  const auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  write(u_h_out, opts.N, opts.N, opts.N * (slice - 1), "results_" + std::to_string(coords[0]));// + "-" + std::to_string(coords[1]));

  // print runtime and exit
  std::cout << "Rank " << rank << ":   2-norm: " << norm_2(sub(b_h, b_h_ref)) << '\n';
  std::cout << "Rank " << rank << ": inf-norm: " << norm_inf(sub(b_h, b_h_ref)) << '\n';
  std::cout << "Rank " << rank << ": Runtime: " << duration.count() << " microseconds" << std::endl;

  // call the MPI final cleanup routine
  MPI_Finalize();

  return EXIT_SUCCESS;
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}