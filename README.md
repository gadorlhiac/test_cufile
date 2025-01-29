# Dummy test program for cuFile performance

## Compilation

```bash
nvcc iotest.cu -lcuda -lcufile -o iotest -std=c++11
```

## Usage
```bash
Usage: [CUFILE_ENV_PATH_JSON="/cufile/config.json"] GDSFILE="/output/path.bin" ./iotest [-b bufSize] [-c] [-h] [-n nThreads] [-N numIter] [-p]
	-b $bufSize   Single buffersize to use. Default: 0x6C000.
	-c            Use cuFile (may be required depending on memory mode)
	-d            Use O_DIRECT file descriptor.
	-h            Display this help message.
	-n $nThreads  Number of threads/streams to use. Default 2.
	-N $numIter   Number of iterations of bufSize writes. Default 100000.
	-p            Use pinned/mapped memory instead of global dev memory.

GDSFILE is an env var that specifies where to write the output to.
CUFILE_ENV_PATH_JSON is a cuFile environment variable that points to a cuFile configuration JSON file.
```

## CUFILE_ENV_PATH_JSON

cuFile parameters are configured via a JSON file. A default may be installed at `/etc/cufile.json`. One is also included here and can be modified and passed to the program using the `CUFILE_ENV_PATH_JSON` environment variable.

**WEKA configuration will likely need to be modified in the JSON file.** Some GDS parameters as well.

Logging can be configured as well.

## Using `gdsio` and `gds_stats` for comparison

NVIDIA ships it's own benchmarking tools as well: `gdsio` and `gds_stats`. These are useful for comparison. `gdsio` is run first to start IO tests, afterwhich `gds_stats` can be run to gather performance statistics on the running process.

### `gdsio` Usage
```bash
Usage [using config file]: gdsio rw-sample.gdsio
Usage [using cmd line options]:./gdsio
         -f <file name>
         -D <directory name>
         -d <gpu_index (refer nvidia-smi)>
         -n <numa node>
         -m <memory type(0 - (cudaMalloc), 1 - (cuMem), 2 - (cudaMallocHost), 3 - (malloc) 4 - (mmap))>
         -w <number of threads for a job>
         -s <file size(K|M|G)>
         -o <start offset(K|M|G)>
         -i <io_size(K|M|G)> <min_size:max_size:step_size>
         -p <enable nvlinks>
         -b <skip bufregister>
         -V <verify IO>
         -x <xfer_type> [0(GPU_DIRECT), 1(CPU_ONLY), 2(CPU_GPU), 3(CPU_ASYNC_GPU), 4(CPU_CACHED_GPU), 5(GPU_DIRECT_ASYNC), 6(GPU_BATCH), 7(GPU_BATCH_STREAM)]
         -B <batch size>
         -I <(read) 0|(write)1| (randread) 2| (randwrite) 3>
         -T <duration in seconds>
         -k <random_seed> (number e.g. 3456) to be used with random read/write>
         -U <use unaligned(4K) random offsets>
         -R <fill io buffer with random data>
         -F <refill io buffer with random data during each write>
         -a <alignment size in case of random IO>
         -P <rdma url>
         -J <per job statistics>

xfer_type:
0 - Storage->GPU (GDS)
1 - Storage->CPU
2 - Storage->CPU->GPU
3 - Storage->CPU->GPU_ASYNC
4 - Storage->PAGE_CACHE->CPU->GPU
5 - Storage->GPU_ASYNC
6 - Storage->GPU_BATCH
7 - Storage->GPU_BATCH_STREAM

Note:
read test (-I 0) with verify option (-V) should be used with files written (-I 1) with -V option
read test (-I 2) with verify option (-V) should be used with files written (-I 3) with -V option, using same random seed (-k),
same number of threads(-w), offset(-o), and data size(-s)
write test (-I 1/3) with verify option (-V) will perform writes followed by read
```

**xfer_type** 0,5,6,7 are most interesting

**When using multiple threads (-w NTHREADS), multiple files will be written**


### Example with `gdsio` and `gds_stats`

```bash
> CUFILE_ENV_PATH_JSON="/path/to/cufile.json" ./gdsio -D /path/to/output/dir -d 0 -w 8 -s 100G -i 1M -x 6 -I 1 -T 100 &
[1] 1588968 # Use this for gds_stats
> watch -n 1 "./gds_stats -p 1588968 | grep BandWidth"
```
