#ifndef DIFFUSION_BASELINE_H_
#define DIFFUSION_BASELINE_H_

#include "diffusion/diffusion.h"

#include <string>

namespace diffusion {

class Baseline: public Diffusion {
 protected:
  REAL *f1_, *f2_;
 public:
  Baseline(int ndim, const int *dims):
      Diffusion(ndim, dims), f1_(NULL), f2_(NULL) {}
  
  virtual std::string GetName() const {
    return std::string("baseline") + std::to_string(ndim_) + std::string("d");
  }
  
  virtual void InitializeBenchmark() {
    f1_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f1_);    
    f2_ = (REAL*)malloc(sizeof(REAL) * nx_ * ny_ * nz_);
    assert(f2_);
    Initialize(f1_, nx_, ny_, nz_,
               kx_, ky_, kz_, dx_, dy_, dz_,
               kappa_, 0.0, ndim_);
  }
  
  virtual void FinalizeBenchmark() {
    assert(f1_);
    free(f1_);
    assert(f2_);
    free(f2_);
  }
    
  virtual void RunKernel(int count) {
    int i;
    for (i = 0; i < count; ++i) {
      int z;
      for (z = 0; z < nz_; z++) {
        int y;
        for (y = 0; y < ny_; y++) {
          int x;
          for (x = 0; x < nx_; x++) {
            int c, w, e, n, s, b, t;
            c =  x + y * nx_ + z * nx_ * ny_;
            w = (x == 0)    ? c : c - 1;
            e = (x == nx_-1) ? c : c + 1;
            n = (y == 0)    ? c : c - nx_;
            s = (y == ny_-1) ? c : c + nx_;
            b = (z == 0)    ? c : c - nx_ * ny_;
            t = (z == nz_-1) ? c : c + nx_ * ny_;
            REAL f = 0;
            if (ndim_ == 2) {
              f =  cc_ * f1_[c] + cw_ * f1_[w] + ce_ * f1_[e]
                  + cs_ * f1_[s] + cn_ * f1_[n];
            } else if (ndim_ == 3) {
              f = cc_ * f1_[c] + cw_ * f1_[w] + ce_ * f1_[e]
                  + cs_ * f1_[s] + cn_ * f1_[n] + cb_ * f1_[b] + ct_ * f1_[t];
            }
            f2_[c] = f;
          }
        }
      }
      REAL *t = f1_;
      f1_ = f2_;
      f2_ = t;
    }
    return;
  }
    
  virtual REAL GetAccuracy(int count) {
    REAL *ref = GetCorrectAnswer(count);
    REAL err = 0.0;
    long len = nx_*ny_*nz_;
    for (long i = 0; i < len; i++) {
      REAL diff = ref[i] - f1_[i];
      err +=  diff * diff;
    }
    return (REAL)sqrt(err/len);
  }
    
  virtual void Dump() const {
    FILE *out = fopen(GetDumpPath().c_str(), "w");
    assert(out);
    long nitems = nx_ * ny_ * nz_;
    for (long i = 0; i < nitems; ++i) {
      fprintf(out, "%.7E\n", f1_[i]);
    }
    fclose(out);
  }
    
};


}

#endif /* DIFFUSION_BASELINE_H_ */
