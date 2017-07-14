#ifndef DIFFUSION_DIFFUSION_OPENMP_H_
#define DIFFUSION_DIFFUSION_OPENMP_H_

#include "diffusion/diffusion.h"
#include "diffusion/baseline.h"

namespace diffusion {

class DiffusionOpenMP: public Baseline {
 public:
  DiffusionOpenMP(int nd, const int *dims):
      Baseline(nd, dims) {}
  virtual std::string GetName() const {
    return std::string("openmp");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);

 protected:
  virtual void Initialize2D();
  virtual void Initialize3D();  
  virtual void RunKernel2D(int count);  
  virtual void RunKernel3D(int count);
  
};

}

#endif /* DIFFUSION_DIFFUSION_OPENMP_H_ */
