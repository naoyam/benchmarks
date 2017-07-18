#ifndef DIFFUSION_DIFFUSION_CUDA_SHARED_H_
#define DIFFUSION_DIFFUSION_CUDA_SHARED_H_

#include "diffusion/diffusion_cuda.h"

namespace diffusion {

class DiffusionCUDAShared1: public DiffusionCUDA {
 public:
  DiffusionCUDAShared1(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared2: public DiffusionCUDA {
 public:
  DiffusionCUDAShared2(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared2");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_OPT_H_
