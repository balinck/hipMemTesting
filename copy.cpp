#include <iostream>
#include <math.h>
#include <string.h>
#include <fstream>
#include <vector>

#include "hip/hip_runtime.h"

__global__
void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

__global__ 
void validateResult(int n, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    assert(y[i] == 3.0f);
  }
}

class MemTest {
  protected:
    hipEvent_t initStart, initStop;
    hipEvent_t copyStart, copyStop;
    hipEvent_t execStart, execStop;
    int blockSize, numBlocks, N;
    float *devX, *devY, *hostX, *hostY;
    std::string name;

  public:
    MemTest(int _N, int _blockSize, int _numBlocks, std::string _name) {
      hipEventCreate(&initStart);
      hipEventCreate(&initStop);
      hipEventCreate(&copyStart);
      hipEventCreate(&copyStop);
      hipEventCreate(&execStart);
      hipEventCreate(&execStop);
      N = _N;
      blockSize = _blockSize;
      numBlocks = _numBlocks;
      name = _name;
    }

    void InitArrays(float *_x, float *_y) {
      for (int i = 0; i < N; i++) {
        _x[i] = 1.0f;
	_y[i] = 2.0f;
      }
    }

    void Exec() {
      hipEventRecord(execStart);
      add<<<numBlocks, blockSize>>>(N, devX, devY);
      hipEventRecord(execStop);
    }

    void Run(std::ofstream *outfile) {
      std::cout << "Test " << name;
      Init();
      Copy();
      Exec();
      hipEventSynchronize(execStop);
      hipDeviceSynchronize();
      std::cout << std::endl << "\t" 
	        << hipGetErrorString(hipGetLastError())
		<< std::endl
      ;
      float initMs, copyMs, execMs;
      hipEventElapsedTime(&initMs, initStart, initStop);
      hipEventElapsedTime(&copyMs, copyStart, copyStop);
      hipEventElapsedTime(&execMs, execStart, execStop);
      Validate();
      hipDeviceSynchronize();
      
      hipFree(devX);
      hipFree(devY);
      hipDeviceSynchronize();
     
      *outfile << name << "," << N << "," << initMs << "," 
	       << copyMs << "," << execMs << "," << std::endl
      ;
    }

    void Validate() {
      validateResult<<<numBlocks, blockSize>>>(N, devY);
    }

    virtual void Init() = 0;
    virtual void Copy() = 0;
};

class ManagedMemory : public MemTest {
  public:
    bool doPrefetching;
    ManagedMemory(int _N,  int _blockSize, int _numBlocks,
		  std::string _name, bool _doPrefetching) 
	    : MemTest(_N, _blockSize, _numBlocks, _name) {
      doPrefetching = _doPrefetching;
    }

    void Init() override {
      hipEventRecord(initStart);
      hipMallocManaged(&devX, N*sizeof(float));
      hipMallocManaged(&devY, N*sizeof(float));
      InitArrays(devX, devY);
      hipEventRecord(initStop);
    }

    void Copy() override {
      hipEventRecord(copyStart);
      if (doPrefetching) {
        hipMemPrefetchAsync(devX, N*sizeof(float), 0);
        hipMemPrefetchAsync(devY, N*sizeof(float), 0);
      }
      hipEventRecord(copyStop);
    }
};

class ExplicitCopy : public MemTest {
  using MemTest::MemTest;
  public:

    void Init() override {
      hipEventRecord(initStart);
      hostX = (float *) malloc(N*sizeof(float));
      hostY = (float *) malloc(N*sizeof(float));
      hipMalloc(&devX, N*sizeof(float));
      hipMalloc(&devY, N*sizeof(float));
      InitArrays(hostX, hostY);
      hipEventRecord(initStop);
    }

    void Copy() override {
      hipEventRecord(copyStart);
      hipMemcpy(devX, hostX, N*sizeof(float), hipMemcpyHostToDevice);
      hipMemcpy(devY, hostY, N*sizeof(float), hipMemcpyHostToDevice);
      hipEventRecord(copyStop);
      free(hostX);
      free(hostY);    
    }
};

class PinnedMemory : public MemTest {
  using MemTest::MemTest;
  public:
    void Init() override {
      hipEventRecord(initStart);
      hipHostMalloc(&hostX, N*sizeof(float), hipHostMallocDefault);
      hipHostMalloc(&hostY, N*sizeof(float), hipHostMallocDefault);
      hipMalloc(&devX, N*sizeof(float));
      hipMalloc(&devY, N*sizeof(float));
      InitArrays(hostX, hostY);
      hipEventRecord(initStop);
    }

    void Copy() override {
      hipEventRecord(copyStart);
      hipMemcpy(devX, hostX, N*sizeof(float), hipMemcpyHostToDevice);
      hipMemcpy(devY, hostY, N*sizeof(float), hipMemcpyHostToDevice);
      hipEventRecord(copyStop);
    }

    ~PinnedMemory() {
      hipHostFree(hostX);
      hipHostFree(hostY);  
    }
};

int main(int argc, char *argv[]) {
  std::ofstream outfile;
  outfile.open("output.csv", std::ios::out);
  
  if (argc < 2) {
    std::cout << "bad arguments, specify number of tests per type" 
	      << std::endl
    ; 
    exit(-1); 
  } 
  int numTests = atoi(argv[1]);
  std::vector<int> sizes = {100000, 1000000, 10000000, 100000000};
  int total = numTests * 4 * sizes.size();

  outfile << "benchmark," << "size," << "initMs,"
	  << "copyMs," << "execMs," << std::endl
  ;

  int N;
  int blockSize;
  int numBlocks;
  int t = 0;
  for (int i = 0; i < sizes.size(); i++) {
    N = sizes[i];
    blockSize = 64;
    numBlocks = (N + blockSize - 1) / blockSize;
    for (int j = 0; j < numTests; j++) {
      std::cout << "(" << ++t << "/" << total << ")" << " ";
      auto test1 = new ManagedMemory(N, blockSize, numBlocks, "managedMem", true);
      test1->Run(&outfile);
      hipDeviceSynchronize();
      delete test1;

      std::cout << "(" << ++t << "/" << total << ")" << " ";
      auto test2 = new ManagedMemory(N, blockSize, numBlocks, "managedMemNoPrefetch", false);
      test2->Run(&outfile);
      hipDeviceSynchronize();
      delete test2;

      std::cout << "(" << ++t << "/" << total << ")" << " ";
      auto test3 = new ExplicitCopy(N, blockSize, numBlocks, "explicitCopy");
      test3->Run(&outfile);
      hipDeviceSynchronize();
      delete test3;

      std::cout << "(" << ++t << "/" << total << ")" << " ";
      auto test4 = new PinnedMemory(N, blockSize, numBlocks, "pinnedMem");
      test4->Run(&outfile);
      hipDeviceSynchronize();
      delete test4;
    }
  }
  return 0;
}
