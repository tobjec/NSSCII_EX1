#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include <mpi.h> // https://www-lb.open-mpi.org/doc/v4.1/

int main(int argc, char** argv) {

  // first thing to do in an MPI program
  MPI_Init(&argc, &argv);

  // obtain own global rank
  int grk;
  MPI_Comm_rank(MPI_COMM_WORLD, &grk);

  // open a (binary) file with shared access to collected non-scabled "cout" from each process for debugging purposes
  // the content of the file will be printed at the end by one process only
  MPI_File shared_cout;
  MPI_File_open(MPI_COMM_WORLD, "shared_cout.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_DELETE_ON_CLOSE,
                MPI_INFO_NULL, &shared_cout);

  // wrapper for blocking output of a string to the shared "cout file"
  auto mpi_print = [&shared_cout](const std::string& str) {
    MPI_Status status;
    return MPI_File_write_shared(shared_cout, str.data(), str.size(), MPI_CHAR, &status);
  };

  // printing a simple string to the shared cout
  std::stringstream ss;
  ss << "hello from global rank:" << grk << std::endl;
  mpi_print(ss.str());

  // obtain number of global MPI processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  { // using a two-dimensional (2D) cartesian topology

    constexpr int n = 2; // one dimension

    // letting MPI find a proper select of the dimensions (obsolete for one-dimension)
    std::array<int, n> dims = {0, 0};
    MPI_Dims_create(size, n, std::data(dims));

    // create a communicator for a 2D topology (with potential reordering of ranks)
    std::array<int, n> periodic = {false, false};
    int reorder = true;
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), reorder, &comm);

    // obtain own rank in 2D communicator
    int rk;
    MPI_Comm_rank(comm, &rk);

    // obtain and store neighboring ranks in the 2D topology (left/right)
    constexpr int displ = 1;
    std::array<int, n * 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    enum Direction : int { LEFT = 0, RIGHT = 1, TOP = 2, BOTTOM = 3 };
    MPI_Cart_shift(comm, 0, displ, &nb[BOTTOM], &nb[TOP]);
    MPI_Cart_shift(comm, 1, displ, &nb[LEFT], &nb[RIGHT]);

    // obtain own coordinates
    std::array<int, n> coord = {-1, -1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    // setup own data domain: 4x4 row-major contiguous data:
    // rk , rk , rk, rk
    // rk , rk , rk, rk
    // rk , rk , rk, rk
    // rk , rk , rk, rk
    constexpr int N = 4;
    std::array<double, N * N> data;
    data.fill(rk);

    // wrapper for printing own data domain
    auto print_data = [&rk, &coord, &data]() {
      std::stringstream cout;
      cout << "coord=(" << coord[0] << "," << coord[1] << ") rank=" << rk << " :" << std::endl;
      for (const auto& col : {0, 1, 2, 3}) {
        cout << "[";
        for (const auto& row : {0, 1, 2, 3})
          cout << " " << data[row + N * col] << " ";
        cout << "]" << std::endl;
      }
      return cout.str();
    };

    // print own data domain before commuincation
    mpi_print(print_data());

    // register MPI data types (here vectors) to send a row or a column
    MPI_Datatype row;
    MPI_Type_vector(N - 2, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Datatype col;
    MPI_Type_vector(N - 2, 1, N, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);

    // send/recv to all four neighbors
    std::array<MPI_Request, n * 2 * 2> req;

    // send own middle column to left; recv the middle column from right
    MPI_Isend(&data[N + 1], 1, col, nb[LEFT], LEFT, comm, &req[0]);
    MPI_Irecv(&data[N + N - 1], 1, col, nb[RIGHT], LEFT, comm, &req[1]);

    // send own middle column to right; recv the middle column from left
    MPI_Isend(&data[N + 1], 1, col, nb[RIGHT], RIGHT, comm, &req[2]);
    MPI_Irecv(&data[N], 1, col, nb[LEFT], RIGHT, comm, &req[3]);

    // send own middle row to bottom; recv the middle row from top
    MPI_Isend(&data[N + 1], 1, row, nb[BOTTOM], BOTTOM, comm, &req[4]);
    MPI_Irecv(&data[N * N - N + 1], 1, row, nb[TOP], BOTTOM, comm, &req[5]);

    // send own middle row to top; recv the middle row from bottom
    MPI_Isend(&data[N + 1], 1, row, nb[TOP], TOP, comm, &req[6]);
    MPI_Irecv(&data[+1], 1, row, nb[BOTTOM], TOP, comm, &req[7]);

    // wait/block for/until all four communication calls to finish
    std::array<MPI_Status, n * 2 * 2> status;
    MPI_Waitall(n * 2 * 2, std::data(req), std::data(status));

    { // print status of each of the four communications for debugging
      std::stringstream cout;
      cout << "rank" << rk << " cout:" << std::endl;
      for (const auto& item : status)
        cout << "status: sender:" << item.MPI_SOURCE << ", tag: " << item.MPI_TAG << ", error code: " << item.MPI_ERROR
             << std::endl;
      mpi_print(cout.str());
    }

    // wrait for all processes in the 1D communicator to reach here
    MPI_Barrier(comm);

    // print own domain data again, now this should look likes this:
    // rk   , TOP    , TOP    , rk
    // LEFT , rk     , rk     , RIGHT
    // LEFT , rk     , rk     , RIGHT
    // rk   , BOTTOM , BOTTOM , rk
    mpi_print(print_data());

    // free 1d communicator
    MPI_Comm_free(&comm);
  }

  // wait for all processes in the global communicator to reach here
  MPI_Barrier(MPI_COMM_WORLD);

  // close and print the shared cout to the console
  if (grk == 0) {
    std::ifstream cout("shared_cout.txt", std::ios::binary);
    std::copy(std::istreambuf_iterator<char>(cout), std::istreambuf_iterator<char>(),
              std::ostreambuf_iterator<char>(std::cout));
    MPI_File_close(&shared_cout);
  } else {
    MPI_File_close(&shared_cout);
  }

  // call the MPI final cleanup routine
  MPI_Finalize();

  return 0;
}
