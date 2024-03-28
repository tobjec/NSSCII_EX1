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

  { // using a one-dimensional (1D) cartesian topology

    constexpr int n = 1; // one dimension

    // letting MPI find a proper select of the dimensions (obsolete for one-dimension)
    std::array<int, n> dims = {0};
    MPI_Dims_create(size, n, std::data(dims));

    // create a communicator for a 1D topology (with potential reordering of ranks)
    std::array<int, n> periodic = {false};
    int reorder = true;
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), reorder, &comm);

    // obtain own rank in 1D communicator
    int rk;
    MPI_Comm_rank(comm, &rk);

    // obtain and store neighboring ranks in the 1D topology (left/right)
    constexpr int displ = 1;
    std::array<int, n* 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL};
    enum Direction : int { LEFT = 0, RIGHT = 1 };
    MPI_Cart_shift(comm, 0, displ, &nb[LEFT], &nb[RIGHT]);

    // obtain own coordinates
    std::array<int, n> coord = {-1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    // setup own data domain: 3x3 row-major contiguous data:
    // rk , rk , rk
    // rk , rk , rk
    // rk , rk , rk
    constexpr int N = 3;
    std::array<double, N * N> data;
    data.fill(rk);

    // wrapper for printing own data domain
    auto print_data = [&rk, &coord, &data]() {
      std::stringstream cout;
      cout << "coord=(" << coord[0] << ",) rank=" << rk << " :" << std::endl;
      for (const auto& col : {0, 1, 2}) {
        cout << "[";
        for (const auto& row : {0, 1, 2})
          cout << " " << data[row + N * col] << " ";
        cout << "]" << std::endl;
      }
      return cout.str();
    };

    // print own data domain before commuincation
    mpi_print(print_data());

    // register MPI data types (here vectors) to send a row or a column
    MPI_Datatype row;
    MPI_Type_vector(N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Datatype col;
    MPI_Type_vector(N, 1, N, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);

    // send own middle column to both neighbors
    // recieve a column from left/right neighbor and store as right/left column
    enum Request : int { LEFT_SEND = 0, RIGHT_SEND = 1, LEFT_RECV = 2, RIGHT_RECV = 3 };
    enum Tag : int { LEFT_Msg = 0, RIGHT_Msg = 1 };
    std::array<MPI_Request, n * 2 * 2> req;
    MPI_Isend(&data[1], 1, col, nb[LEFT], LEFT_Msg, comm, &req[LEFT_SEND]);
    MPI_Isend(&data[1], 1, col, nb[RIGHT], RIGHT_Msg, comm, &req[RIGHT_SEND]);
    MPI_Irecv(&data[0], 1, col, nb[LEFT], RIGHT_Msg, comm, &req[LEFT_RECV]);
    MPI_Irecv(&data[2], 1, col, nb[RIGHT], LEFT_Msg, comm, &req[RIGHT_RECV]);

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
    // rk-1 , rk , rk+1
    // rk-1 , rk , rk+1
    // rk-1 , rk , rk+1
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
