#define USE_LDG
#include "diffusion/diffusion_cuda_shfl.h"
#include "common/cuda_util.h"


namespace diffusion {
namespace cuda_shfl2 {

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
    int x = 0;    
    x_offset = 0;
    REAL fw = __shfl_up(fc[x], 1);
    if (tid == 0 && i != 0) {
      fw = f1[p-1];
    }
    REAL fe = __shfl_down(fc[x], 1);
    REAL fe_b = __shfl(fc[x+1], 0);
    fe = (tid == WARP_SIZE -1) ? fe_b : fe;
    f2[p+x_offset] = cc * fc[x] + cw * fw + ce * fe
        + cs * fs[x] + cn * fn[x];
    x_offset += WARP_SIZE; 

    
    PRAGMA_UNROLL
    for (int x = 1; x < NUM_WB_X-1; ++x) {
      REAL fw = __shfl_up(fc[x], 1);
      REAL fw_b = __shfl(fc[x-1], WARP_SIZE - 1);
      fw = (tid == 0) ? fw_b : fw;
      
      REAL fe = __shfl_down(fc[x], 1);
      REAL fe_b = __shfl(fc[x+1], 0);
      fe = (tid == WARP_SIZE -1) ? fe_b : fe;
      
      f2[p+x_offset] = cc * fc[x] + cw * fw + ce * fe
          + cs * fs[x] + cn * fn[x];
      x_offset += WARP_SIZE; 
    }

    // last boundary
    x = NUM_WB_X - 1;
    fw = __shfl_up(fc[x], 1);
    REAL fw_b = __shfl(fc[x-1], WARP_SIZE - 1);
    fw = (tid == 0) ? fw_b : fw;
    fe = __shfl_down(fc[x], 1);
    if (tid == WARP_SIZE -1) {
      if (i + x_offset != nx - 1) {
        fe = f1[p+x_offset+1];
      }
    }
      
    f2[p+x_offset] = cc * fc[x] + cw * fw + ce * fe
        + cs * fs[x] + cn * fn[x];
    x_offset += WARP_SIZE; 

    
    p += nx;
  }
  return;
}

} // namespace cuda_shfl2

void DiffusionCUDASHFL2::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  dim3 block_dim(WARP_SIZE, 1);
  dim3 grid_dim(nx_ / BLOCK_X, ny_ / BLOCK_Y);
  if (ndim_ == 3) grid_dim.z = grid_z_;
  
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      cuda_shfl2::kernel2d<<<grid_dim, block_dim>>>
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

void DiffusionCUDASHFL2::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl2::kernel2d,
                                          cudaFuncCachePreferL1));
}



} // namespace diffusion
