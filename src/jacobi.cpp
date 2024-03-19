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

using namespace std::chrono;

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

struct Index_2D
{
  constexpr Index_2D(std::size_t N) : N{N}, h{1. / (N - 1.)} {}
  constexpr std::size_t operator()(std::size_t i, std::size_t j) const { return j * N + i; }
  constexpr double x(std::size_t i) const { return i * h; }
  constexpr double y(std::size_t j) const { return j * h; }
  const std::size_t N;
  const double h;
};

void gaussian_source_region(const double source_x, const double source_y, const double source_sigma, std::vector<double>& b_h, const Index_2D i2d) {
  for (auto i = 0; i < i2d.N; ++i) {
    for (auto j = 0; j < i2d.N; ++j) {
      const auto x_diff = i2d.x(i) - source_x, y_diff = i2d.y(j) - source_y;
      b_h[i2d(i, j)] = 1 / (2 * M_PI * source_sigma * source_sigma) * std::exp(-(x_diff * x_diff + y_diff * y_diff / 2 / source_sigma / source_sigma));
    }
  }
}

void jacobi_iterate(const std::vector<double>& u_h, std::vector<double>& u_h_next, const std::vector<double>& b_h, const Index_2D i2d, double fixed_west, double fixed_east) {
  for (auto i = 0; i < i2d.N; ++i) {
    for (auto j = 0; j < i2d.N; ++j) {
      const auto u_C =                 u_h[i2d(i    , j    )];
      const auto u_W = i > 0         ? u_h[i2d(i - 1, j    )] : fixed_west;
      const auto u_E = i < i2d.N - 1 ? u_h[i2d(i + 1, j    )] : fixed_east;
      const auto u_S = j > 0         ? u_h[i2d(i    , j - 1)] : u_C;
      const auto u_N = j < i2d.N - 1 ? u_h[i2d(i    , j + 1)] : u_C;

      u_h_next[i2d(i, j)] = i2d.h * i2d.h / 4. * (b_h[i2d(i, j)] + 1. / i2d.h / i2d.h * (u_W + u_E + u_S + u_N));
    }
  }
}

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

// compute A_h * u_h = b_h with A_h given implicitly by the stencil and boundary conditions
void recompute(const std::vector<double>& u_h, std::vector<double>& b_h, const Index_2D i2d, double fixed_west, double fixed_east) {
  for (auto i = 0; i < i2d.N; ++i) {
    for (auto j = 0; j < i2d.N; ++j) {
      const auto u_C =                 u_h[i2d(i    , j    )];
      const auto u_W = i > 0         ? u_h[i2d(i - 1, j    )] : fixed_west;
      const auto u_E = i < i2d.N - 1 ? u_h[i2d(i + 1, j    )] : fixed_east;
      const auto u_S = j > 0         ? u_h[i2d(i    , j - 1)] : u_C;
      const auto u_N = j < i2d.N - 1 ? u_h[i2d(i    , j + 1)] : u_C;

      b_h[i2d(i, j)] = -1. / i2d.h / i2d.h * (u_W + u_E + u_S + u_N - 4 * u_C);
    }
  }
}

// solve A_h * u_h = b_h with A_h given implicitly by the stencil and boundary conditions
void solve(std::vector<double>& u_h_in, std::vector<double>& u_h_out, const std::vector<double>& b_h, double fix_west, double fix_east, const Index_2D i2d, std::size_t iters) {
  // use zero-vector as initial solution
  std::fill(u_h_in.begin(), u_h_in.end(), 0.);

  for (auto iter = 0; iter < iters; ++iter) {
    jacobi_iterate(u_h_in, u_h_out, b_h, i2d, fix_west, fix_east);

    // The next iteration's input is the current iteration's output
    std::swap(u_h_in, u_h_out);
  }
}

auto run(program_options::Options opt) {
  const auto i2d = Index_2D{opt.N};

  const auto grid_size = i2d.N * i2d.N;
  std::vector<double> u_h_in(grid_size), u_h_out(grid_size), b_h(grid_size);

  if (opt.has_source)
    gaussian_source_region(opt.source_x, opt.source_y, opt.source_sigma, b_h, i2d);
  else
    std::fill(b_h.begin(), b_h.end(), 0.);

  solve(u_h_in, u_h_out, b_h, opt.fix_west, opt.fix_east, i2d, opt.iters);

  std::vector<double> b_h_ref(grid_size);
  recompute(u_h_out, b_h_ref, i2d, opt.fix_west, opt.fix_east);

  const auto diff = sub(b_h, b_h_ref);

  std::cout << "  2-norm: " << norm_2(diff) << '\n';
  std::cout << "inf-norm: " << norm_inf(diff) << '\n';

  return u_h_out;
}

int main(int argc, char *argv[]) try {
  const auto start = std::chrono::high_resolution_clock::now();
  auto opts = program_options::parse(argc, argv);

  opts.print();

  // write csv
  auto write = [ N = opts.N, name = opts.name ](const auto &x) -> auto {
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

  const auto solution = run(opts);
  write(solution);
  const auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Runtime: " << duration.count() << " microseconds" << std::endl;

  return EXIT_SUCCESS;
} catch (std::exception &e) {
  std::cout << e.what() << std::endl;
  return EXIT_FAILURE;
}