#include <cufile.h>
#include "cuda.h"

#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <unistd.h>
#include <thread>
#include <string>
#include <vector>
#include <getopt.h>

/**
 * Basic test program for cuFile and posix IO throughput.
 * Compile with:
 * nvcc iotest.cu -lcuda -lcufile -o iotest -std=c++11
 *
 * Does not appear to work with NFS mounts of weka file systems.
 * Lustre via NFS seems to work. As do "normal" local file systems and wekafs
 * directly.
 */

/// For cudaError_t directly

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line)
{
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line)
{
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

/// For dealing with the errors in cuFile structs

int checkCuda(CUfileError_t status)
{
  if (IS_CUDA_ERR(status)) {
    const char *str;
    CUresult strResult = cuGetErrorString(status.cu_err, &str);
    if (strResult == CUDA_SUCCESS) {
      std::cout << "CUDA ERROR: " << str << std::endl;
      return -2;
    } else {
      return -3;
    }
  }
  return 0;
}

int checkCuFile(CUfileError_t status)
{
  if (IS_CUFILE_ERR(status.err)) {
    std::cout << "cuFile ERROR " << status.err << ": "
              << CUFILE_ERRSTR(status.err) << std::endl;
    if (status.err == 5011)
      checkCuda(status);
    return -1;
  }
  return checkCuda(status);
}

struct GPUTimer {
  cudaEvent_t beg, end;
  GPUTimer() {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }
  ~GPUTimer() {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }
  void start() { cudaEventRecord(beg, 0); }
  double stop() {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return ms;
  }
};

/*                                Data organization
 * |--------|--------|--------|--------| |--------|--------|--------|--------|
 *     S0       S1       S2       S3     ...     S0       S1       S2       S3
 *                   E0                                        En
 */

/**
 * /param fd File descriptor.
 * /param fd_flags Flags to use to open descriptor.
 * /param stream CUDA stream to use.
 * /param tid Thread id. This is used to calculate offsets for writing the file.
 * /param nThreads Total number of threads. Used to calculate offsets for writes.
 * /param bufSize Number of bytes to write.
 * /param memThroughput Memory write throughput pointer.
 * /param ioThroughput IO write throughput pointer.
 * /param nIter Number of iterations to run. For averaging better statistics. Like "events"
 */

void write_posix_file_pinned_mem(const char *filename,
                                 const int fd_flags,
                                 cudaStream_t stream,
                                 size_t tid,
                                 size_t nThreads,
                                 size_t bufSize,
                                 double *memThroughput,
                                 double *ioThroughput,
                                 size_t nIter)
{
  std::cout << "TID " << tid << " started." << std::endl;

  int fd = open(filename, fd_flags, 0644);
  if (fd < 0) {
    std::cout << "Unable to open file: " << errno << std::endl;
    abort();
  } else {
    std::cout << "Successfully created: " << filename << " with fd: " << fd
              << std::endl;
  }

  void* hostPtr;
  void* devPtr;

  CHECK_CUDA_ERROR(cudaHostAlloc(&hostPtr, bufSize, cudaHostAllocMapped));
  cudaStreamSynchronize(0);
  std::cout << "TID " << tid << ": allocated HOST buffer of size: " << bufSize << std::endl;

  CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&devPtr, hostPtr, 0));

  std::cout << "TID " << tid << ": Retrieved device ptr of mapped HOST memory." << std::endl;

  size_t memVal = 0xde + tid;

  size_t eventOffset; // Offset in file for an "event" all streams write a full event
  off_t fileOffset; // Offset in file for the stream within a single event
  ssize_t bytesWritten {0};

  //GPUTimer memTimer;
  GPUTimer ioTimer;
  GPUTimer writeTimer;
  //memTimer.start();
  //for (size_t i=0; i<nIter; ++i)
  //  CHECK_CUDA_ERROR(cudaMemsetAsync(devPtr, memVal, bufSize, stream));
  CHECK_CUDA_ERROR(cudaMemsetAsync(devPtr, memVal, bufSize, stream));
  cudaStreamSynchronize(stream);
  //double memTime = memTimer.stop(); // in ms
  //*memThroughput += (bufSize*nIter / 1e9) / (memTime * 0.001);
  ioTimer.start();
  for (size_t i=0; i < nIter; ++i) {
    eventOffset = i*nThreads*bufSize;
    fileOffset = eventOffset + bufSize * tid;

    lseek(fd, fileOffset, SEEK_SET);
    bytesWritten = write(fd, hostPtr, bufSize);
  }
  double ioTime = ioTimer.stop(); // in ms
  *ioThroughput += (bytesWritten*nIter / 1e9) / (ioTime * 0.001);

  CHECK_CUDA_ERROR(cudaFreeHost(hostPtr));
  std::cout << "Freed HOST buffer." << std::endl;

  close(fd);
  std::cout << "TID " << tid << ": File closed." << std::endl;
}

/**
 * /brief Test cuFile write throughput and memory write throughput using pinned memory.
 * /param fd File descriptor.
 * /param fd_flags Flags to use to open descriptor.
 * /param stream CUDA stream to use.
 * /param tid Thread id. This is used to calculate offsets for writing the file.
 * /param nThreads Total number of threads. Used to calculate offsets for writes.
 * /param bufSize Number of bytes to write.
 * /param memThroughput Memory write throughput pointer.
 * /param ioThroughput IO write throughput pointer.
 * /param nIter Number of iterations to run. For averaging better statistics. Like "events"
 */

void write_cufile_file_pinned_mem(const char* filename,
                                  const int fd_flags,
                                  cudaStream_t stream,
                                  size_t tid,
                                  size_t nThreads,
                                  size_t bufSize,
                                  double* memThroughput,
                                  double* ioThroughput,
                                  size_t nIter)
{
  std::cout << "TID " << tid << " started." << std::endl;

  int fd = open(filename, fd_flags, 0644);
  if (fd < 0) {
    std::cout << "Unable to open file: " << errno << std::endl;
    abort();
  } else {
    std::cout << "Successfully created: " << filename << " with fd: " << fd
              << std::endl;
  }

  CUfileDescr_t descr;
  CUfileHandle_t handle;

  memset(reinterpret_cast<void *>(&descr), 0, sizeof(CUfileDescr_t));

  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  if (checkCuFile(cuFileHandleRegister(&handle, &descr))) {
    std::cout << "Error registering file handle for fd " << fd << std::endl;
    close(fd);
    abort();
  } else {
    std::cout << "cuFile handle registered" << std::endl;
  }

  void* hostPtr;
  void* devPtr;

  CHECK_CUDA_ERROR(cudaHostAlloc(&hostPtr, bufSize, cudaHostAllocMapped));
  cudaStreamSynchronize(0);
  std::cout << "TID " << tid << ": allocated HOST buffer of size: " << bufSize << std::endl;

  CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&devPtr, hostPtr, 0));

  std::cout << "TID " << tid << ": Retrieved device ptr of mapped HOST memory." << std::endl;

  if (checkCuFile(cuFileBufRegister(devPtr, bufSize, 0))) {
    std::cout << "Unable to register GPU buffer with cuFile." << std::endl;
    cuFileHandleDeregister(handle);
    CHECK_CUDA_ERROR(cudaFreeHost(hostPtr));
    abort();
  } else {
    std::cout << "TID " << tid << ": GPU buffer registered with cuFile." << std::endl;
  }

  size_t memVal = 0xde + tid;

  size_t eventOffset; // Offset in file for an "event" all streams write a full event
  off_t fileOffset; // Offset in file for the stream within a single event
  off_t devPtrOffset;
  ssize_t bytesWritten {0};


  //GPUTimer memTimer;
  GPUTimer ioTimer;
  //memTimer.start();
  //for (size_t i=0; i <nIter; ++i) {
  //  CHECK_CUDA_ERROR(cudaMemsetAsync(devPtr, memVal, bufSize, stream));
  //}
  CHECK_CUDA_ERROR(cudaMemsetAsync(devPtr, memVal, bufSize, stream));
  cudaStreamSynchronize(stream);
  //double memTime = memTimer.stop(); // in ms
  //*memThroughput += (bufSize * nIter / 1e9) / (memTime * 0.001);
  ioTimer.start();
  for (size_t i=0; i < nIter; ++i) {
    eventOffset = i*nThreads*bufSize;
    fileOffset = eventOffset + bufSize * tid;

    if (checkCuFile(cuFileWriteAsync(handle, devPtr, &bufSize, &fileOffset,
                                     &devPtrOffset, &bytesWritten, stream))) {
      std::cout << "TID " << tid << ": saw error on cuFileWrite." << std::endl;
    }
  }
  cudaStreamSynchronize(stream);
  double ioTime = ioTimer.stop(); // in ms
  *ioThroughput += (bytesWritten*nIter / 1e9) / (ioTime * 0.001);

  if (checkCuFile(cuFileBufDeregister(devPtr))) {
    std::cout << "cuFile deregister GPU buffer failed!" << std::endl;
    CHECK_CUDA_ERROR(cudaFreeHost(hostPtr));
    cuFileHandleDeregister(handle);
    abort();
  }

  CHECK_CUDA_ERROR(cudaFreeHost(hostPtr));
  std::cout << "Freed HOST buffer." << std::endl;

  cuFileHandleDeregister(handle);
  std::cout << "TID " << tid << ": De-registered cuFile handle." << std::endl;
}



/**
 * /brief Test cuFile write throughput and memory write throughput using standard global memory.
 * /param filename Filename to open.
 * /param fd_flags Flags to use to open descriptor.
 * /param stream CUDA stream to use.
 * /param tid Thread id. This is used to calculate offsets for writing the file.
 * /param nThreads Total number of threads. Used to calculate offsets for writes.
 * /param bufSize Number of bytes to write.
 * /param memThroughput Memory write throughput pointer.
 * /param ioThroughput IO write throughput pointer.
 * /param nIter Number of iterations to run. For averaging better statistics. Like "events"
 */

void write_cufile_file_global_mem(const char* filename,
                                  const int fd_flags,
                                  cudaStream_t stream,
                                  size_t tid,
                                  size_t nThreads,
                                  size_t bufSize,
                                  double* memThroughput,
                                  double* ioThroughput,
                                  size_t nIter)
{
  std::cout << "TID " << tid << " started." << std::endl;

  int fd = open(filename, fd_flags, 0644);
  if (fd < 0) {
    std::cout << "Unable to open file: " << errno << std::endl;
    abort();
  } else {
    std::cout << "Successfully created: " << filename << " with fd: " << fd
              << std::endl;
  }

  CUfileDescr_t descr;
  CUfileHandle_t handle;

  memset(reinterpret_cast<void *>(&descr), 0, sizeof(CUfileDescr_t));

  descr.handle.fd = fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  if (checkCuFile(cuFileHandleRegister(&handle, &descr))) {
    std::cout << "Error registering file handle for fd " << fd << std::endl;
    close(fd);
    abort();
  } else {
    std::cout << "cuFile handle registered" << std::endl;
  }

  void* devPtr;

  CHECK_CUDA_ERROR(cudaMallocAsync(&devPtr, bufSize, stream));
  cudaStreamSynchronize(stream);
  std::cout << "TID " << tid << ": allocated GPU buffer of size: " << bufSize << std::endl;


  if (checkCuFile(cuFileBufRegister(devPtr, bufSize, 0))) {
    std::cout << "Unable to register GPU buffer with cuFile." << std::endl;
    cuFileHandleDeregister(handle);
    cudaFree(devPtr);
    abort();
  } else {
    std::cout << "TID " << tid << ": GPU buffer registered with cuFile." << std::endl;
  }

  size_t memVal = 0xde + tid;

  size_t eventOffset; // Offset in file for an "event" all streams write a full event
  off_t fileOffset; // Offset in file for the stream within a single event
  off_t devPtrOffset;
  ssize_t bytesWritten = 0;

  //GPUTimer memTimer;
  GPUTimer ioTimer;
  //memTimer.start();
  //for (size_t i= 0; i < nIter; ++i) {
  //  CHECK_CUDA_ERROR(cudaMemsetAsync(devPtr, memVal, bufSize, stream));
  //}
  CHECK_CUDA_ERROR(cudaMemsetAsync(devPtr, memVal, bufSize, stream));
  cudaStreamSynchronize(stream);
  //double memTime = memTimer.stop(); // in ms
  //*memThroughput = (bufSize*nIter / 1e9) / (memTime * 0.001);
  ioTimer.start();
  for (size_t i=0; i < nIter; ++i) {
    eventOffset = i*nThreads*bufSize;
    fileOffset = eventOffset + bufSize * tid;

    if (checkCuFile(cuFileWriteAsync(handle, devPtr, &bufSize, &fileOffset,
                                     &devPtrOffset, &bytesWritten, stream))) {
      std::cout << "TID " << tid << ": saw error on cuFileWrite." << std::endl;
    }
  }
  cudaStreamSynchronize(stream);
  double ioTime = ioTimer.stop(); // in ms
  *ioThroughput = (bytesWritten*nIter / 1e9) / (ioTime * 0.001);

  if (bytesWritten < 0 || bytesWritten != bufSize) {
    std::cout << "cuFileWrite failed!" << std::endl;
  } else {
    std::cout << "TID " << tid << ": cuFileWrite succesful" << std::endl;
  }


  if (checkCuFile(cuFileBufDeregister(devPtr))) {
    std::cout << "cuFile deregister GPU buffer failed!" << std::endl;
    cudaFree(devPtr);
    cuFileHandleDeregister(handle);
    abort();
  }

  CHECK_CUDA_ERROR(cudaFree(devPtr));
  std::cout << "Freed GPU buffer." << std::endl;

  cuFileHandleDeregister(handle);
  std::cout << "TID " << tid << ": De-registered cuFile handle." << std::endl;
}

void help_msg(char* program)
{
  std::cout
      << "Usage: [CUFILE_ENV_PATH_JSON=\"/cufile/config.json\"] GDSFILE=\"/output/path\" "
      << program << " [-b bufSize] [-c] [-h] [-n nThreads] [-N numIter] [-p]"      << std::endl
      << "\t-b $bufSize   Single buffersize to use. Default: 0x6C000."             << std::endl
      << "\t-c            Use cuFile (may be required depending on memory mode)"   << std::endl
      << "\t-d            Use O_DIRECT file descriptor."                           << std::endl
      << "\t-h            Display this help message."                              << std::endl
      << "\t-n $nThreads  Number of threads/streams to use. Default 2."            << std::endl
      << "\t-N $numIter   Number of iterations of bufSize writes. Default 100000." << std::endl
      << "\t-p            Use pinned/mapped memory instead of global dev memory."  << std::endl
      << std::endl
      << "GDSFILE is an env var that specifies where to write the output to."      << std::endl
      << "CUFILE_ENV_PATH_JSON is a cuFile environment variable that points to a"
      << " cuFile configuration JSON file." << std::endl;
}

int main(int argc, char* argv[])
{

  char c;
  size_t bufSize = sizeof(float) * 192 * 144 * 4;
  bool usecuFile = false;
  bool useDirectIO = false;
  size_t nThreads = 2;
  size_t nIter = 100000;
  bool usePinnedMemory = false;
  while ((c = getopt(argc, argv, "b:cdhn:N:p")) != -1) {
    switch (c) {
    case 'b':
      bufSize = std::atoi(optarg);
      break;
    case 'c':
      usecuFile = true;
      break;
    case 'd':
      useDirectIO = true;
      break;
    case 'h':
      help_msg(argv[0]);
      exit(0);
    case 'n':
      nThreads = std::atoi(optarg);
      break;
    case 'N':
      nIter = std::atoi(optarg);
      break;
    case 'p':
      usePinnedMemory = true;
      break;
    }
  }

  if (!usecuFile && !usePinnedMemory) {
    std::cout << "Requested no cuFile and non-pinned/mapped memory. This choice is incompatible."
              << std::endl
              << "Will switch to using pinned/mapped memory." << std::endl;
    usePinnedMemory = true;
  }


  std::cout << "Running IO/memory throughput test with the following options: " << std::endl
            << "Buffer size: " << bufSize << std::endl
            << "Number of threads and streams: " << nThreads << std::endl
            << "Number of iterations (# buffers): " << nIter << std::endl;
  if (usecuFile)
    std::cout << "Using cuFile." << std::endl;
  else
    std::cout << "Using POSIX write." << std::endl;
  if (usePinnedMemory)
    std::cout << "Using pinned/mapped memory." << std::endl;
  else
    std::cout << "Using standard device global memory." << std::endl;

  //cudaSetDevice(0);

  if (usecuFile) {
    if (checkCuFile(cuFileDriverOpen())) {
      std::cout << "Error on initializing cuFile infrastructure." << std::endl;
      abort();
    }
  }

  std::vector<std::thread> threads;
  cudaStream_t streams[nThreads];

  for (size_t i=0; i < nThreads; ++i) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    if (usecuFile) {
      if (checkCuFile(cuFileStreamRegister(streams[i],  0))) {
        std::cout << "cuFile unable to register cudaStream_t " << i << std::endl;
      } else {
        std::cout << "cuFile registered cudaStream_t " << i << std::endl;
      }
    }
  }

  if (usecuFile) {
    CUfileDrvProps_t devProps;
    if (checkCuFile(cuFileDriverGetProperties(&devProps))) {
      std::cout << "Cannot read cuFile capabilities." << std::endl;
      abort();
    } else {
      std::cout << "cuFile major version: "   << devProps.nvfs.major_version      << std::endl
                << "cuFile minor version: "   << devProps.nvfs.minor_version      << std::endl
                << "cuFile poll thresh size:" << devProps.nvfs.poll_thresh_size   << std::endl
                << "cuFile max dir. IO size:" << devProps.nvfs.max_direct_io_size << std::endl
                << "cuFile dstatus flags: "   << devProps.nvfs.dstatusflags       << std::endl
                << "cuFile dcontrol flags: "  << devProps.nvfs.dcontrolflags      << std::endl;

      std::cout << "cuFile max device cache size: "   << devProps.max_device_cache_size      << std::endl
                << "cuFile per buffer cache size: "   << devProps.per_buffer_cache_size      << std::endl
                << "cuFile max pin. memory size: "    << devProps.max_device_pinned_mem_size << std::endl
                << "cuFile max batch io timeout ms: " << devProps.max_batch_io_timeout_msecs << std::endl
                << "cuFile max batch io: "            << devProps.max_batch_io_size          << std::endl;
    }
  }

  char cwd[1024];
  if (!getcwd(cwd, sizeof(cwd))) {
    std::cout << "Cannot get current working directory!" << std::endl;
    abort();
  } else {
    std::cout << "Working in: " << cwd << std::endl;
  }
  const char* filename = getenv("GDSFILE");
  if (filename == NULL) {
    std::cout << "Must provide a base filename through GDSFILE environment variable!" << std::endl;
    abort();
  }

  int fd_flags = O_CREAT | O_WRONLY;
  if (useDirectIO)
    fd_flags |= O_DIRECT;

  double memTimePerThread[nThreads];
  double ioTimePerThread[nThreads];

  GPUTimer throughputTimer;

  for (size_t t = 0; t < nThreads; ++t) {
    if (usecuFile) {
      if (!usePinnedMemory) {
        threads.push_back(std::thread(write_cufile_file_global_mem,
                                      filename, fd_flags, streams[t], t,
                                      nThreads, bufSize, memTimePerThread+t,
                                      ioTimePerThread+t, nIter));
      } else {
        threads.push_back(std::thread(write_cufile_file_pinned_mem,
                                      filename, fd_flags, streams[t], t,
                                      nThreads, bufSize, memTimePerThread+t,
                                      ioTimePerThread + t, nIter));
      }
    } else {
        threads.push_back(std::thread(write_posix_file_pinned_mem,
                                      filename, fd_flags, streams[t], t,
                                      nThreads, bufSize, memTimePerThread+t,
                                      ioTimePerThread+t, nIter));
    }
  }
  throughputTimer.start();
  for (auto &t : threads) {
    t.join();
  }

  // totalTime measures more than just IO/mem writes. But it's okay as a proxy...
  double totalTime = throughputTimer.stop();
  if (usecuFile) {
    if (checkCuFile(cuFileDriverClose())) {
      std::cout << "Error on finalizing the cuFile system." << std::endl;
      abort();
    } else {
      std::cout << "cuFile driver closed." << std::endl;
    }
  }
  for (size_t s=0; s < nThreads; ++s) {
    cudaStreamDestroy(streams[s]);
  }

  double memThroughput{0};
  double ioThroughput{0};
  double totalThroughput{(bufSize*nIter*nThreads/1e9)/(totalTime*0.001)};
  for (size_t i=0; i<nThreads; ++i) {
    ioThroughput += ioTimePerThread[i];
    memThroughput += memTimePerThread[i];
  }


  //std::cout << "Memory throughput per thread is: " << memThroughput/nThreads << " GB/s" << std::endl
  std::cout << "IO throughput per thread is: " << ioThroughput/nThreads << " GB/s" << std::endl;

  std::cout << "Total throughput is: " << totalThroughput << " GB/s" << std::endl;
}
