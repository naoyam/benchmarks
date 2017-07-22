#define USE_LDG
#include "diffusion/diffusion_cuda_shfl.h"
#include "common/cuda_util.h"

#define WARP_SIZE (32)
#define WARP_MASK (WARP_SIZE-1)
#define NUM_WB_X (BLOCK_X / WARP_SIZE)

namespace diffusion {
namespace cuda_shfl1 {

#if 0
__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  int j_end = j + BLOCK_Y;

  int p = OFFSET2D(i, j, nx);

  REAL fn, fc, fs;
  fn = f1[p];
  fc = (blockIdx.y > 0) ? f1[p-nx] : fn;

  for (; j < j_end; ++j) {
    SHIFT3(fs, fc, fn);
    fn = (j < ny - 1) ? f1[p+nx] : fn;

    // shfl
#if 1
    REAL fw = __shfl_up(fc, 1);
    if (tid == 0 && i > 0) fw = f1[p-1];
    REAL fe = __shfl_down(fc, 1);
    if (tid == WARP_SIZE -1 && i + 1 < nx) fe = f1[p+1];
#else
    REAL fw = 0;
    REAL fe = 0;
#endif
    
    f2[p] = cc * fc + cw * fw + ce * fe
        + cs * fs + cn * fn;
    p += nx;
  }

  return;
}
#else
__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  int tid = threadIdx.x;
  int i = BLOCK_X * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  int j_end = j + BLOCK_Y;

  int p = OFFSET2D(i, j, nx);

  REAL fn[NUM_WB_X], fc[NUM_WB_X], fs[NUM_WB_X];

  for (int x = 0; x < NUM_WB_X; ++x) {
    fn[x] = f1[p+x*WARP_SIZE];
    fc[x] = (blockIdx.y > 0) ? f1[p+x*WARP_SIZE-nx] : fn[x];
  }
  
  for (; j < j_end; ++j) {
    int x_offset = 0;
    // loads in batch
    PRAGMA_UNROLL
    for (int x = 0; x < NUM_WB_X; ++x) {
      SHIFT3(fs[x], fc[x], fn[x]);
      fn[x] = (j < ny - 1) ? f1[p+x_offset+nx] : fn[x];
      x_offset += WARP_SIZE;      
    }

    // compute
    x_offset = 0;
    PRAGMA_UNROLL
    for (int x = 0; x < NUM_WB_X; ++x) {
      REAL fw = __shfl_up(fc[x], 1);
      REAL fw_prev_warp = 0;
      if (x > 0) fw_prev_warp = __shfl(fc[x-1], WARP_SIZE - 1);
      if (tid == 0) {
        if (x == 0) {
          if (i != 0) {
            fw = f1[p-1];
          }
        } else {
          fw = fw_prev_warp;
        }
      }
      REAL fe = __shfl_down(fc[x], 1);
      REAL fe_next_warp = 0;
      if (x < NUM_WB_X-1) fe_next_warp = __shfl(fc[x+1], 0);
      if (tid == WARP_SIZE -1) {
        if (x == NUM_WB_X - 1) {
          if (i + x_offset != nx - 1) {
            fe = f1[p+x_offset+1];
          }
        } else {
          fe = fe_next_warp;
        }
      }
      
      f2[p+x_offset] = cc * fc[x] + cw * fw + ce * fe
          + cs * fs[x] + cn * fn[x];
      x_offset += WARP_SIZE; 
    }

    p += nx;
  }
  return;
}
#endif

} // namespace cuda_shfl1

void DiffusionCUDASHFL1::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(WARP_SIZE, 1);
  dim3 grid_dim(nx_ / BLOCK_X, ny_ / BLOCK_Y);
  if (ndim_ == 3) grid_dim.z = grid_z_;
  
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      cuda_shfl1::kernel2d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
    } else if (ndim_ == 3) {
      assert(0);
      //cuda_shfl1::kernel3d<<<grid_dim, block_dim>>>
      //(f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);
    }
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDASHFL1::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl1::kernel2d,
                                          cudaFuncCachePreferL1));
}



} // namespace diffusion
