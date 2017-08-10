#define USE_LDG
#include "diffusion/diffusion_cuda_shfl_temp.h"
#include "common/cuda_util.h"

#define USE_LDG

#define BLOCK_T (2)
#define BLOCK_T_MASK ((BLOCK_T)-1)

namespace diffusion {
namespace cuda_shfl_temp1 {

#define STENCIL2D(fc, fn, fs, fe, fw) \
  (cc * (fc) + cw * (fw) + ce * (fe) + cs * (fs) + cn * (fn))

#define RCIDX(y) ((y)&1)
#define RSIDX(y) (RCIDX(y)^1)

__device__ REAL load(F1_DECL f1, int x, int y,
                     int nx, int ny) {
  x = MIN(MAX(x, 0), nx - 1);
  y = MIN(MAX(y, 0), ny - 1);
  return f1[x+nx*y];
}

#define KERNEL_COMP(t, r0, r1) do {                             \
    if (yt < 0) {                                               \
      r[t][0] = fn;                                             \
      b[t][0] = bn;                                             \
      --yt;                                                     \
      break;                                                    \
    } else if (yt >= ny) {                                      \
      --yt;                                                     \
      continue;                                                 \
    }                                                           \
                                                                \
    REAL fc = r[t][r0];                                         \
    if (t == 0) {                                               \
      fn = load(f1, i, j + yt + 1, nx, ny);                     \
    } else if (yt == ny - 1) {                                  \
      fn = fc;                                                  \
    }                                                           \
    REAL fs;                                                    \
    if (yt == 0) {                                              \
      fs = fc;                                                  \
    } else {                                                    \
      fs = r[t][r1];                                            \
    }                                                           \
    REAL bc = b[t][r0];                                         \
    if (t == 0) {                                               \
      bn = load(f1, ib, j + yt + 1, nx, ny);                    \
    } else if (yt == ny - 1) {                                  \
      bn = bc;                                                  \
    }                                                           \
    REAL bs;                                                    \
    if (yt == 0) {                                              \
      bs = bc;                                                  \
    } else {                                                    \
      bs = b[t][r1];                                            \
    }                                                           \
    REAL fw = __shfl(fc, (tid - 1 + WARP_SIZE) & WARP_MASK);    \
    REAL fe = __shfl(fc, (tid + 1) & WARP_MASK);                \
    REAL bw = __shfl(bc, (tid - 1 + WARP_SIZE) & WARP_MASK);    \
    REAL be = __shfl(bc, (tid + 1) & WARP_MASK);                \
    if (tid == 0) {                                             \
      SWAP(fw, bw, REAL);                                       \
    } else if (tid == WARP_SIZE - 1) {                          \
      SWAP(fe, be, REAL);                                       \
    }                                                           \
    if (i == 0) {                                               \
      fw = fc;                                                  \
    } else if (i == nx - 1) {                                   \
      fe = fc;                                                  \
    }                                                           \
    REAL f_next = STENCIL2D(fc, fn, fs, fe, fw);                \
    r[t][r1] = fn;                                              \
    fn = f_next;                                                \
    REAL b_next = STENCIL2D(bc, bn, bs, be, bw);                \
    b[t][r1] = bn;                                              \
    bn = b_next;                                                \
    --yt;                                                       \
  } while (0)

__global__ void kernel2d(F1_DECL f1, F2_DECL f2,
                         int nx, int ny,
                         REAL ce, REAL cw, REAL cn, REAL cs,
                         REAL cc) {
  // this should have no effect to the final result, but may tell the
  //compiler tid is less than WARP_SIZE
  //int tid = threadIdx.x;
  int tid = threadIdx.x & WARP_MASK;
  int i = BLOCK_X * blockIdx.x + threadIdx.x;
  int ib;
  if (tid < (WARP_SIZE / 2)) {
    ib = BLOCK_X * blockIdx.x + WARP_SIZE + (tid & BLOCK_T_MASK);
  } else {
    ib = BLOCK_X * blockIdx.x - WARP_SIZE
        + (tid | ((WARP_SIZE - 1) & (~BLOCK_T_MASK)));
  }

  int j = BLOCK_Y * blockIdx.y;
  //int j_end = j + BLOCK_Y;

  int p = OFFSET2D(i, j, nx);

  REAL r[BLOCK_T][2];
  REAL b[BLOCK_T][2];

  int x = 0;
  int x_offset = 0;
  
  r[0][0] = f1[p]; // load(f1, i, j, nx, ny);
  b[0][0] = load(f1, ib, j, nx, ny);

  for (int y = 0; y < ny + BLOCK_T - 1; y += 2) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      REAL fn;
      REAL bn;
      int yt = y + k;
      int r0 = 0;
      int r1 = 1;

#if BLOCK_T == 1
      KERNEL_COMP(0, r0, r1);
#elif BLOCK_T == 2
      KERNEL_COMP(0, r0, r1);
      KERNEL_COMP(1, r1, r0);      
#elif BLOCK_T == 4
      KERNEL_COMP(0, r0, r1);
      KERNEL_COMP(1, r1, r0);      
      KERNEL_COMP(2, r0, r1);
      KERNEL_COMP(3, r1, r0);      
#endif
      ++yt;
      if (yt >= 0) f2[p+nx*yt] = fn;
    }
  }

  if ((ny + BLOCK_T - 1) % 2 == 1) {
    int y = ny + BLOCK_T - 2;
    REAL fn;
    REAL bn;
    int yt = y;
    int r0 = 0;
    int r1 = 1;

#if BLOCK_T == 1
    KERNEL_COMP(0, r0, r1);
#elif BLOCK_T == 2
    KERNEL_COMP(0, r0, r1);
    KERNEL_COMP(1, r1, r0);      
#elif BLOCK_T == 4
    KERNEL_COMP(0, r0, r1);
    KERNEL_COMP(1, r1, r0);      
    KERNEL_COMP(2, r0, r1);
    KERNEL_COMP(3, r1, r0);      
#endif
    ++yt;
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
  
  assert(count % BLOCK_T == 0);
  
  CHECK_CUDA(cudaEventRecord(ev1_));
  for (int i = 0; i < count; i += BLOCK_T) {
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
