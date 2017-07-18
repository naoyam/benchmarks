#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <string>
#include <vector>
#include <map>


using std::vector;
using std::string;
using std::map;
using std::make_pair;

#include "diffusion/diffusion.h"
#include "diffusion/baseline.h"
#if defined(OPENMP)
#include "diffusion/diffusion_openmp.h"
#endif

#if defined(CUDA) || defined(CUDA_L1)
#include "diffusion/diffusion_cuda.h"
#endif
#ifdef CUDA_ROC
#include "diffusion/diffusion_cuda_roc.h"
#endif
#if defined(CUDA_OPT1) || defined(CUDA_OPT2)
#include "diffusion/diffusion_cuda_opt.h"
#endif

#if 0
#if defined(OPENMP_TEMPORAL_BLOCKING)
#include "diffusion3d/diffusion3d_openmp_temporal_blocking.h"
#endif


#if defined(CUDA_TEMPORAL_BLOCKING)
#include "diffusion/diffusion_cuda_temporal_blocking.h"
#endif

#if defined(MIC)
#include "diffusion/diffusion_mic.h"
#endif

#if defined(PHYSIS)
#include "diffusion/diffusion_physis.h"
#endif

#if defined(FORTRAN) || defined(FORTRAN_ACC)
#include "diffusion/diffusion_fortran.h"
#endif
#endif

using namespace diffusion;
using std::string;


static const int COUNT = 100;
static const int ND = 3;
static const int SIZE = 256;

void Die() {
  std::cerr << "FAILED!!!\n";
  exit(EXIT_FAILURE);
}

void PrintUsage(std::ostream &os, char *prog_name) {
  os << "Usage: " << prog_name << " [options] [benchmarks]\n\n";
  os << "Options\n"
     << "\t--nd N      " << "Number of dimensions (default: " << ND << ")\n"
     << "\t--count N   " << "Number of iterations (default: " << COUNT << ")\n"
     << "\t--size N    "  << "Size of each dimension (default: " << SIZE << ")\n"
     << "\t--dump      "  << "Dump the final data to file\n"
     << "\t--help      "  << "Display this help message\n";
}


void ProcessProgramOptions(int argc, char *argv[],
                           int &nd, int &count, int &size,
                           bool &dump) {
  int c;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
      {"nd", 1, 0, 0},
      {"count", 1, 0, 0},
      {"size", 1, 0, 0},
      {"dump", 0, 0, 0},
      {"help", 0, 0, 0},      
      {0, 0, 0, 0}
    };

    c = getopt_long(argc, argv, "",
                    long_options, &option_index);
    if (c == -1) break;
    if (c != 0) {
      //std::cerr << "Invalid usage\n";
      //PrintUsage(std::cerr, argv[0]);
      //Die();
      continue;
    }

    switch(option_index) {
      case 0:
        nd = atoi(optarg);
        break;
      case 1:
        count = atoi(optarg);
        break;
      case 2:
        size = atoi(optarg);
        break;
      case 3:
        dump = true;
        break;
      case 4:
        PrintUsage(std::cerr, argv[0]);
        exit(EXIT_SUCCESS);
        break;
      default:
        break;
    }
  }
}

int main(int argc, char *argv[]) {
  int nd = ND;
  int size = SIZE; // default size
  int count = COUNT; // default iteration count
  bool dump = false;
  
  ProcessProgramOptions(argc, argv, nd, count, size, dump);

  assert(nd >= 2 && nd <= 3);
  
  Diffusion *bmk = NULL;

  std::vector<int> dims(nd, size);

#if defined(OPENMP)
  bmk = new DiffusionOpenMP(nd, dims.data());
#elif defined(OPENMP_TEMPORAL_BLOCKING)
  bmk = new DiffusionOpenMPTemporalBlocking(nx, nx, nx);
#elif defined(CUDA) || defined(CUDA_L1)
  bmk = new DiffusionCUDA(nd, dims.data());
#elif defined(CUDA_ROC)
  bmk = new DiffusionCUDAROC(nd, dims.data());
#elif defined(CUDA_ZBLOCK)
  bmk = new DiffusionCUDAZBlock(nx, nx, nx);
#elif defined(CUDA_OPT1)
  bmk = new DiffusionCUDAOpt1(nd, dims.data());
#elif defined(CUDA_OPT2)
  bmk = new DiffusionCUDAOpt2(nd, dims.data());
#elif defined(CUDA_OPT3)
  bmk = new DiffusionCUDAOpt3(nx, nx, nx);
#elif defined(CUDA_SHARED)
  bmk = new DiffusionCUDAShared(nx, nx, nx);
#elif defined(CUDA_SHARED1)
  bmk = new DiffusionCUDAShared1(nx, nx, nx);
#elif defined(CUDA_SHARED2)
  bmk = new DiffusionCUDAShared2(nx, nx, nx);
#elif defined(CUDA_SHARED3)
  bmk = new DiffusionCUDAShared3(nx, nx, nx);
#elif defined(CUDA_SHARED4)
  bmk = new DiffusionCUDAShared4(nx, nx, nx);
#elif defined(CUDA_SHARED5)
  bmk = new DiffusionCUDAShared5(nx, nx, nx);
#elif defined(CUDA_SHARED6)
  bmk = new DiffusionCUDAShared6(nx, nx, nx);
#elif defined(CUDA_XY)
  bmk = new DiffusionCUDAXY(nx, nx, nx);
#elif defined(CUDA_TEMPORAL_BLOCKING)
  bmk = new DiffusionCUDATemporalBlocking(nx, nx, nx);
#elif defined(MIC)
  bmk = new DiffusionMIC(nx, nx, nx);
#elif defined(PHYSIS)
  bmk = new DiffusionPhysis(nx, nx, nx, argc, argv);
#elif defined(FORTRAN) || defined(FORTRAN_ACC)
  bmk = new DiffusionFortran(nx, nx, nx);  
#else
  bmk = new Baseline(nd, dims.data());
#endif
  
  bmk->RunBenchmark(count, dump);

  return 0;
}
