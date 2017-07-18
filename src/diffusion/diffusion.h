#ifndef DIFFUSION_DIFFUSION_H_
#define DIFFUSION_DIFFUSION_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <string>

#define REAL float
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#include "common/stopwatch.h"

#define OFFSET2D(i, j, nx) \
  ((i) + (j) * (nx))
#define OFFSET3D(i, j, k, nx, ny) \
  ((i) + (j) * (nx) + (k) * (nx) * (ny))

#define STRINGIFY(x) #x
#ifndef UNROLL
//#define PRAGMA_UNROLL(x)
#define PRAGMA_UNROLL
#elif UNROLL == 0
//#define PRAGMA_UNROLL(x) _Pragma("unroll")
#define PRAGMA_UNROLL _Pragma("unroll")
#elif UNROLL > 0
#define PRAGMA_UNROLL__(x, y) STRINGIFY(x y)
#define PRAGMA_UNROLL_(x) PRAGMA_UNROLL__(unroll, x)
//#define PRAGMA_UNROLL(x) _Pragma(PRAGMA_UNROLL_(x))
#define PRAGMA_UNROLL _Pragma(PRAGMA_UNROLL_(UNROLL))
#else
#error Invalid macro definition
#endif



namespace diffusion {

inline
void Initialize(REAL *buff, const int nx, const int ny, const int nz,
                const REAL kx, const REAL ky, const REAL kz,
                const REAL dx, const REAL dy, const REAL dz,
                const REAL kappa, const REAL time,
                int ndim) {
  REAL ax = exp(-kappa*time*(kx*kx));
  REAL ay = exp(-kappa*time*(ky*ky));
  REAL az = exp(-kappa*time*(kz*kz));
  int jz;  
  for (jz = 0; jz < nz; jz++) {
    int jy;
    for (jy = 0; jy < ny; jy++) {
      int jx;
      for (jx = 0; jx < nx; jx++) {
        int j = jz*nx*ny + jy*nx + jx;
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
            *(1.0 - ay*cos(ky*y));
        if (ndim == 3) {
          REAL z = dz*((REAL)(jz + 0.5));          
          f0 *= (1.0 - az*cos(kz*z));
        }
        buff[j] = f0;
      }
    }
  }
}
  

class Diffusion {
 protected:
  int ndim_;  
  int nx_;
  int ny_;
  int nz_;
  REAL kappa_;
  REAL dx_, dy_, dz_;
  REAL kx_, ky_, kz_;
  REAL dt_;
  REAL ce_, cw_, cn_, cs_, ct_, cb_, cc_;
  
 public:
  Diffusion(int ndim, const int *dims):
      ndim_(ndim), kappa_(0.1) {
    nx_ = dims[0];
    ny_ = (ndim > 1) ? dims[1] : 1;
    nz_ = (ndim > 2) ? dims[2] : 1;
    REAL l = 1.0;
    dx_ = l / nx_;
    dy_ = l / ny_;
    dz_ = l / nz_;
    kx_ = ky_ = kz_ = 2.0 * M_PI;
    dt_ = 0.1 * dx_ * dx_ / kappa_;
    ce_ = cw_ = kappa_*dt_/(dx_*dx_);
    cn_ = cs_ = kappa_*dt_/(dy_*dy_);
    ct_ = cb_ = kappa_*dt_/(dz_*dz_);
    cc_ = 1.0 - (ce_ + cw_ + cn_ + cs_ + (ndim_ == 3 ? (ct_ + cb_) : 0));
  }
  
  virtual std::string GetName() const = 0;
  void RunBenchmark(int count, bool dump) {
    std::cout << "*** Diffusion Benchmark ***\n";
    std::cout << "Benchmark: " << GetName() << "\n";    
    std::cout << "Initializing benchmark input...\n";
    InitializeBenchmark();
    std::cout << "Iteration count: " << count << "\n";
    std::cout << "Grid size: ";
    if (ndim_ == 2) {
      std::cout << nx_ << "x" << ny_ << "\n";
    } else if (ndim_ == 3) {
      std::cout << nx_ << "x" << ny_ << "x" << nz_ << "\n";
    }
    Stopwatch st;
    StopwatchStart(&st);
    RunKernel(count);
    float elapsed_time = StopwatchStop(&st);
    std::cout << "Benchmarking finished.\n";
    DisplayResult(count, elapsed_time);
    if (dump) Dump();
    FinalizeBenchmark();
  }

 protected:
  std::string GetDumpPath() const {
    return std::string("diffusion_result.")
        + GetName() + std::string(".out");
  }
  virtual void InitializeBenchmark() = 0;  
  virtual void RunKernel(int count) = 0;
  virtual void Dump() const = 0;
  virtual REAL GetAccuracy(int count) = 0;
  virtual void FinalizeBenchmark() = 0;    
  
  float GetThroughput(int count, float time) {
    return (nx_ * ny_ * nz_) * sizeof(REAL) * 2.0 * ((float)count)
        / time * 1.0e-09;    
  }
  float GetGFLOPS(int count, float time) {
    float f = (nx_*ny_*nz_)*13.0*(float)(count)/time * 1.0e-09;
    return f;
  }
  virtual void DisplayResult(int count, float time) {
    printf("Elapsed time : %.3f (s)\n", time);
    printf("FLOPS        : %.3f (GFLOPS)\n",
           GetGFLOPS(count, time));
    printf("Throughput   : %.3f (GB/s)\n",
           GetThroughput(count ,time));
    printf("Accuracy     : %e\n", GetAccuracy(count));
  }
  REAL *GetCorrectAnswer(int count) const {
    REAL *f = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f);
    Initialize(f, nx_, ny_, nz_,
               kx_, ky_, kz_, dx_, dy_, dz_,
               kappa_, count * dt_, ndim_);
    return f;
  }
};

}

#endif /* DIFFUSION_DIFFUSION_H_ */
