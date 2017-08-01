#ifndef DIFFUSION_DIFFUSION_CUDA_SHFL_TEMP_H_
#define DIFFUSION_DIFFUSION_CUDA_SHFL_TEMP_H_

#include "diffusion/diffusion_cuda.h"

#define WARP_SIZE (32)
#define WARP_MASK (WARP_SIZE-1)
#define NUM_WB_X (BLOCK_X / WARP_SIZE)

namespace diffusion {

class DiffusionCUDASHFLTemp1: public DiffusionCUDA {
 public:
  DiffusionCUDASHFLTemp1(int nd, const int *dims):
      DiffusionCUDA(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("cuda_shfl_temp1");
  }
  virtual std::string GetDescription() const {
    return std::string("Data sharing with register shuffling with temporal blocking");
  }
  virtual void Setup();  
  virtual void RunKernel(int count);
  
};

} // namespace diffusion

#endif // DIFFUSION_DIFFUSION_CUDA_SHFL_TEMP_H_
