#include <array>
#include <cstdio>
#include <iostream>
#include <mpi.h> // https://www.open-mpi.org/doc/current/
#include <vector>

#define CHARBUF_SIZE 1024

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    std::cout << "total number of MPI processes=" << num_proc << ":"
              << std::endl;

  if (rank == 0)
    std::cout << "num_proc:" << num_proc << std::endl;

  MPI_Datatype mpi_charbuf;
  MPI_Type_vector(CHARBUF_SIZE, 1, 1, MPI_CHAR, &mpi_charbuf);
  MPI_Type_commit(&mpi_charbuf);

  int ndims = 2;
  std::array<int, 2> dims = {0, 0};
  MPI_Dims_create(num_proc, ndims, std::data(dims));

  if (rank == 0)
    std::printf("2d cart size=(%i,%i)\n", dims[0], dims[1]);

  std::array<int, 2> periods = {false, false};
  int reorder = false;
  MPI_Comm comm_2d;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, std::data(dims), std::data(periods),
                  reorder, &comm_2d);

  int displacement = 1;
  enum DIR : int { N = 0, S = 1, W = 2, E = 3 };
  std::array<int, 4> nb = {-1, -1, -1, -1};
  MPI_Cart_shift(comm_2d, 0, 1, &nb[DIR::N], &nb[DIR::S]);
  MPI_Cart_shift(comm_2d, 1, 1, &nb[DIR::W], &nb[DIR::E]);
  std::array<int, 2> coord = {-1, -1};
  MPI_Cart_coords(comm_2d, rank, ndims, std::data(coord));

  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<char> sendbuf(CHARBUF_SIZE);
  std::sprintf(std::data(sendbuf), "rank=%i, coord=(%i,%i), nb=(%i,%i,%i,%i)",
               rank, coord[0], coord[1], nb[0], nb[1], nb[2], nb[3]);

  // std::cout << std::string_view(std::data(sendbuf)) << std::endl;

  int sendcount = 1;
  int recvcount = 1;
  int recvrank = 0;
  std::vector<char> recvbuf;

  if (rank == recvrank)
    recvbuf.resize(num_proc * CHARBUF_SIZE);

  MPI_Gather(std::data(sendbuf), sendcount, mpi_charbuf, std::data(recvbuf),
             recvcount, mpi_charbuf, recvrank, comm_2d);

  if (rank == recvrank)
    for (int n = 0; n < num_proc; ++n)
      std::cout << std::string_view(std::data(recvbuf) + n * CHARBUF_SIZE)
                << std::endl;

  MPI_Finalize();

  return 0;
}
