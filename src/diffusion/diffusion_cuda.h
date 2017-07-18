#ifndef DIFFUSION_DIFFUSION_CUDA_H_
#define DIFFUSION_DIFFUSION_CUDA_H_

#include "diffusion/diffusion.h"
#include "diffusion/baseline.h"

#include <cuda_runtime.h>
#ifndef BLOCK_X
#define BLOCK_X (64)
#endif
#ifndef BLOCK_Y
#define BLOCK_Y (4)
#endif
#ifndef GRID_Z
#define GRID_Z (4)
#endif

#if defined(ENABLE_ROC)
#define F1_DECL const REAL * __restrict__ f1
#else
#define F1_DECL REAL *f1
#endif

namespace diffusion {

class DiffusionCUDA: public Baseline {
 public:
  DiffusionCUDA(int nd, const int *dims):
      Baseline(nd, dims), f1_d_(NULL), f2_d_(NULL),
      block_x_(BLOCK_X), block_y_(BLOCK_Y), block_z_(1), grid_z_(GRID_Z)
  {
    //assert(nx_ % block_x_ == 0);
    //assert(ny_ % block_y_ == 0);
    //assert(nz_ % block_z_ == 0);
  }
  virtual std::string GetName() const {
    return std::string("cuda");
  }
  virtual void InitializeBenchmark();
  virtual void RunKernel(int count);
  virtual void FinalizeBenchmark();
  virtual void DisplayResult(int count, float time);
  
 protected:
  REAL *f1_d_, *f2_d_;
  int block_x_, block_y_, block_z_;
  int grid_z_;
  cudaEvent_t ev1_, ev2_;

};

#if 0
class DiffusionCUDAZBlock: public DiffusionCUDA {
 public:
  DiffusionCUDAZBlock(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_zblock");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAOpt0: public DiffusionCUDA {
 public:
  DiffusionCUDAOpt0(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_opt0");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAOpt1: public DiffusionCUDA {
 public:
  DiffusionCUDAOpt1(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {}
  virtual std::string GetName() const {
    return std::string("cuda_opt1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAOpt2: public DiffusionCUDA {
 public:
  DiffusionCUDAOpt2(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {
    // block_x_ = 128;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt2");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class DiffusionCUDAOpt3: public DiffusionCUDA {
 public:
  DiffusionCUDAOpt3(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {
    // block_x_ = 128;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_opt3");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class DiffusionCUDAXY: public DiffusionCUDA {
 public:
  DiffusionCUDAXY(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {
    // block_x_ = 32;    
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_xy");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared: public DiffusionCUDA {
 public:
  DiffusionCUDAShared(int nx, int ny, int nz):
      DiffusionCUDA(nx, ny, nz) {
    // block_x_ = 128;
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared");
  }
  virtual void InitializeBenchmark();    
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared1: public DiffusionCUDAShared {
 public:
  DiffusionCUDAShared1(int nx, int ny, int nz):
      DiffusionCUDAShared(nx, ny, nz) {
    // block_x_ = 128;
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared1");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared2: public DiffusionCUDAShared {
 public:
  DiffusionCUDAShared2(int nx, int ny, int nz):
      DiffusionCUDAShared(nx, ny, nz) {
    // block_x_ = 128;
    // block_y_ = 2;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared2");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared3: public DiffusionCUDAShared {
 public:
  DiffusionCUDAShared3(int nx, int ny, int nz):
      DiffusionCUDAShared(nx, ny, nz) {
    // block_x_ = 32;
    // block_y_ = 4;
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared3");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared4: public DiffusionCUDAShared {
 public:
  DiffusionCUDAShared4(int nx, int ny, int nz):
      DiffusionCUDAShared(nx, ny, nz) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared4");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared5: public DiffusionCUDAShared {
 public:
  DiffusionCUDAShared5(int nx, int ny, int nz):
      DiffusionCUDAShared(nx, ny, nz) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared5");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};

class DiffusionCUDAShared6: public DiffusionCUDAShared {
 public:
  DiffusionCUDAShared6(int nx, int ny, int nz):
      DiffusionCUDAShared(nx, ny, nz) {
  }
  virtual std::string GetName() const {
    return std::string("cuda_shared6");
  }
  virtual void InitializeBenchmark();  
  virtual void RunKernel(int count);
};
#endif
}

#endif /* DIFFUSION_DIFFUSION_CUDA_H_ */
