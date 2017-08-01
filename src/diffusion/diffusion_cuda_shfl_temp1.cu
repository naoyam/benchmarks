#define USE_LDG
#include "diffusion/diffusion_cuda_shfl_temp.h"
#include "common/cuda_util.h"

#define USE_LDG

#define BLOCK_T (4)

namespace diffusion {
namespace cuda_shfl_temp1 {

#define STENCIL2D(fc, fn, fs, fe, fw) \
  (cc * (fc) + cw * (fw) + ce * (fe) + cs * (fs) + cn * (fn))

#define RCIDX(y) ((y)&1)
#define RSIDX(y) ((~y)&1)

__device__ REAL load(F1_DECL f1, int p, int y,
                     int nx, int ny) {
  y = MIN(0, y);
  y = MAX(ny-1, y);
  return f1[p+nx*y];
}

__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  // this should have no effect to the final result, but may tell the
  //compiler tid is less than WARP_SIZE
  //int tid = threadIdx.x;
  int tid = threadIdx.x & WARP_MASK;
  int i = BLOCK_X * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  int j_end = j + BLOCK_Y;

  int p = OFFSET2D(i, j, nx);

  REAL r[BLOCK_T][2];
  REAL fe, fw = 0;

  r[0][0] = f1[p];

  for (int y = 0; y < ny + BLOCK_T - 1; ++y) {
    REAL fn;
    int yt = y;
    for (int t = 0; t < BLOCK_T; ++t) {
      if (yt < 0) {
        r[t][0] = fn;
        break;
      } else if (yt >= ny) {
        --yt;
        continue;
      }
      REAL fc = r[t][RCIDX(yt)];
      if (t == 0) {
        fn = load(f1, p, yt+1, nx, ny);
      } else if (yt == ny - 1) {
        fn = fc;
      }
      REAL fs;
      if (yt == 0) {
        fs = fc;
      } else {
        fs = r[t][RSIDX(yt)];
      }
      REAL rn = STENCIL2D(fc, fn, fs, fe, fw);
      r[t][RSIDX(yt)] = fn;
      fn = rn;
      --yt;
    }
    if (yt >= 0) f2[p+nx*yt] = fn;
  }
  
  return;
}

#if 0
__global__ void kernel3d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny, int nz,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL ct, REAL cb, REAL cc) {
  const int tid = threadIdx.x;
  int i = BLOCK_X * blockIdx.x + threadIdx.x;  
  int j = BLOCK_Y * blockIdx.y;
  const int block_z = nz / gridDim.z;
  int k = block_z * blockIdx.z;
  const int k_end = k + block_z;
  const int xy = nx * ny;
  
  int p = OFFSET3D(i, j, k, nx, ny);

  REAL t1[NUM_WB_X][BLOCK_Y+2], t2[NUM_WB_X][BLOCK_Y+2],
      t3[NUM_WB_X][BLOCK_Y+2];

  for (int y = 0; y < BLOCK_Y; ++y) {
    for (int x = 0; x < NUM_WB_X; ++x) {
      t3[x][y+1] = f1[p+x*WARP_SIZE+y*nx];
      t2[x][y+1] = (k > 0) ? f1[p+x*WARP_SIZE+y*nx-xy] : t3[x][y+1];
    }
  }
  {
    int y = -1;
    for (int x = 0; x < NUM_WB_X; ++x) {
      t3[x][y+1] = blockIdx.y == 0 ? t3[x][y+1+1] :
          f1[p+x*WARP_SIZE+y*nx];
    }
  }
  {
    int y = BLOCK_Y;
    for (int x = 0; x < NUM_WB_X; ++x) {
      t3[x][y+1] = blockIdx.y == gridDim.y - 1 ? 
          t3[x][y+1-1] : f1[p+x*WARP_SIZE+y*nx];
    }
  }

  for (; k < k_end; ++k) {
    // load
    PRAGMA_UNROLL    
    for (int y = 0; y < BLOCK_Y; ++y) {
      PRAGMA_UNROLL
      for (int x = 0; x < NUM_WB_X; ++x) {
        SHIFT3(t1[x][y+1], t2[x][y+1], t3[x][y+1]);
        t3[x][y+1] = (k < nz - 1) ? f1[p+x*WARP_SIZE+y*nx+xy]
            : t3[x][y+1];
      }
    }

    int y = -1;
    PRAGMA_UNROLL    
    for (int x = 0; x < NUM_WB_X; ++x) {
      SHIFT3(t1[x][y+1], t2[x][y+1], t3[x][y+1]);      
      if (blockIdx.y == 0) {
        t3[x][y+1] = t3[x][y+1+1];
      } else {
        t3[x][y+1] = (k < nz - 1) ? f1[p+x*WARP_SIZE+y*nx+xy]
            : t3[x][y+1];
      }
    }

    y = BLOCK_Y;
    PRAGMA_UNROLL        
    for (int x = 0; x < NUM_WB_X; ++x) {
      SHIFT3(t1[x][y+1], t2[x][y+1], t3[x][y+1]);      
      if (blockIdx.y == gridDim.y - 1) {
        t3[x][y+1] = t3[x][y+1-1];
      } else {
        t3[x][y+1] = (k < nz - 1) ? f1[p+x*WARP_SIZE+y*nx+xy]
            : t3[x][y+1];
      }
    }

    PRAGMA_UNROLL
    for (int y = 1; y < BLOCK_Y+1; ++y) {
      {
        int x = 0;
        REAL tw = __shfl_up(t2[x][y], 1);
        if (tid == 0 && blockIdx.x > 0) {
          tw = LDG(f1+ p-1+(y-1)*nx);
        }
        REAL te_self = tid == 0 ? t2[x+1][y] :  t2[x][y];
        REAL te = __shfl(te_self, (tid+1) & WARP_MASK);
        f2[p+x*WARP_SIZE+(y-1)*nx] = cc * t2[x][y] + cw * tw
            + ce * te + cs * t2[x][y-1] + cn * t2[x][y+1]
            + cb * t1[x][y] + ct * t3[x][y];
      }
      PRAGMA_UNROLL      
      for (int x = 1; x < NUM_WB_X-1; ++x) {
        REAL tw_self = tid == WARP_SIZE - 1 ? t2[x-1][y] : t2[x][y];
        REAL tw = __shfl(tw_self, (tid-1) & WARP_MASK);
        REAL te_self = tid == 0 ? t2[x+1][y] :  t2[x][y];        
        REAL te = __shfl(te_self, (tid+1) & WARP_MASK);
        f2[p+x*WARP_SIZE+(y-1)*nx] = cc * t2[x][y] + cw * tw
            + ce * te + cs * t2[x][y-1] + cn * t2[x][y+1]
            + cb * t1[x][y] + ct * t3[x][y];
      }
      {
        int x = NUM_WB_X - 1;
        REAL tw_self = tid == WARP_SIZE - 1 ? t2[x-1][y] : t2[x][y];        
        REAL tw = __shfl(tw_self, (tid-1) & WARP_MASK);
        REAL te = __shfl_down(t2[x][y], 1);
        if (tid == WARP_SIZE - 1) {
          if (blockIdx.x < gridDim.x - 1) {
            te = LDG(f1 + p+x*WARP_SIZE+1+(y-1)*nx);
          }
        }
        f2[p+x*WARP_SIZE+(y-1)*nx] = cc * t2[x][y] + cw * tw
            + ce * te + cs * t2[x][y-1] + cn * t2[x][y+1]
            + cb * t1[x][y] + ct * t3[x][y];
      }
    }
    p += xy;
  }
}
#endif

} // namespace cuda_shfl_temp

void DiffusionCUDASHFLTemp1::RunKernel(int count) {
  size_t s = sizeof(REAL) * nx_ * ny_ * nz_;  
  FORCE_CHECK_CUDA(cudaMemcpy(f1_d_, f1_, s, cudaMemcpyHostToDevice));

  assert(BLOCK_X % WARP_SIZE == 0);
  //assert(BLOCK_X / WARP_SIZE >= 2);

  //assert(BLOCK_Y > 1);

  dim3 block_dim(WARP_SIZE, 1);
  //dim3 grid_dim(nx_ / BLOCK_X, ny_ / BLOCK_Y);
  dim3 grid_dim(nx_ / BLOCK_X, 1);
  if (ndim_ == 3) grid_dim.z = grid_z_;

  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; ++i) {
    if (ndim_ == 2) {
      cuda_shfl_temp1::kernel2d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, ce_, cw_, cn_, cs_, cc_);
#if 0      
    } else if (ndim_ == 3) {
      cuda_shfl2::kernel3d<<<grid_dim, block_dim>>>
          (f1_d_, f2_d_, nx_, ny_, nz_, ce_, cw_, cn_, cs_,
           ct_, cb_, cc_);
#endif      
    }
    REAL *t = f1_d_;
    f1_d_ = f2_d_;
    f2_d_ = t;
  }
  CHECK_CUDA(cudaEventRecord(ev2_));
  FORCE_CHECK_CUDA(cudaMemcpy(f1_, f1_d_, s, cudaMemcpyDeviceToHost));
  return;
}

void DiffusionCUDASHFLTemp1::Setup() {
  DiffusionCUDA::Setup();
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl_temp1::kernel2d,
                                          cudaFuncCachePreferL1));
#if 0  
  FORCE_CHECK_CUDA(cudaFuncSetCacheConfig(cuda_shfl_temp2::kernel3d,
                                          cudaFuncCachePreferL1));
#endif  
}



} // namespace diffusion
