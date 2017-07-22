#ifndef DIFFUSION_DIFFUSION_CUDA_SHFL_H_
#define DIFFUSION_DIFFUSION_CUDA_SHFL_H_

#include "diffusion/diffusion_cuda.h"

namespace diffusion {

class DiffusionCUDASHFL1: public DiffusionCUDA {
 public:
  DiffusionCUDASHFL1(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("cuda_shfl1");
  }
  virtual std::string GetDescription() const {
    return std::string("Data sharing with register shuffling");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
  
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_SHFL_H_