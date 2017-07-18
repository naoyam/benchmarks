#ifndef DIFFUSION_DIFFUSION_CUDA_OPT_H_
#define DIFFUSION_DIFFUSION_CUDA_OPT_H_

#include "diffusion/diffusion_cuda.h"

namespace diffusion {

class DiffusionCUDAOpt1: public DiffusionCUDA {
 public:
  DiffusionCUDAOpt1(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // This version optimizes 3D stencils. No 2D version is
    // available.
    assert(nd == 3);
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAOpt2: public DiffusionCUDA {
 public:
  DiffusionCUDAOpt2(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {
    // This version optimizes 3D stencils. No 2D version is
    // available.
    assert(nd == 3);
    // block_x_ = 128;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt2");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_OPT_H_
