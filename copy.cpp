#include <iostream>
#include <math.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

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

void getOpt(char ** begin, char ** end, std::string opt, int* out) {
  char ** itr = std::find(begin, end, opt);

  if (itr != end && ++itr != end) {
    *out = atoi(*itr);
  }
}

class MemTest {
  protected:
    hipEvent_t initStart, initStop;
    hipEvent_t copyStart, copyStop;
    hipEvent_t execStart, execStop;
    int blockSize, numBlocks;
    float *devX, *devY, *hostX, *hostY;

  public:
    std::string name;
    int N;
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
      std::cout << "Test " << name << " size=" << N;
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
	       << copyMs << "," << execMs << "," << -1 << "," << std::endl
      ;
    }

    void Validate() {
      validateResult<<<numBlocks, blockSize>>>(N, devY);
    }

    virtual void Init() = 0;
    virtual void Copy() = 0;
    virtual MemTest* Clone() = 0;

    virtual ~MemTest() {

    }
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
	int dev;
	hipGetDevice(&dev);
        hipMemPrefetchAsync(devX, N*sizeof(float), dev);
        hipMemPrefetchAsync(devY, N*sizeof(float), dev);
      }
      hipEventRecord(copyStop);
    }

    ManagedMemory* Clone() override {
      return new ManagedMemory(*this);
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

    ExplicitCopy* Clone() override {
      return new ExplicitCopy(*this);
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

    PinnedMemory* Clone() override {
      return new PinnedMemory(*this);
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
 
  int dev = 0;
  getOpt(argv, argv+argc, "--dev", &dev);

  if (hipSetDevice(dev) != hipSuccess) {
    std::cout << "invalid deviceId: " << dev << std::endl;
    exit(-1);
  }  

  int numTests = atoi(argv[1]);
  std::vector<int> sizes = {100000, 1000000, 10000000, 100000000};
  int total = numTests * 4 * sizes.size();

  outfile << "benchmark," << "size," << "initMs,"
	  << "copyMs," << "execMs," << "clock," << std::endl
  ;

  int N;
  int blockSize;
  int numBlocks;
  std::vector<MemTest*> testConfigs;
  for (int i = 0; i < sizes.size(); i++) {
    N = sizes[i];
    blockSize = warpSize;
    numBlocks = (N + blockSize - 1) / blockSize;

    testConfigs.push_back(
		    new ManagedMemory(N, blockSize, 
			              numBlocks, "managedMem", 
				      true)
    );

    testConfigs.push_back(
		    new ManagedMemory(N, blockSize, 
			              numBlocks, "managedMemNoPrefetch",
				      false)
    );

    testConfigs.push_back(
		    new ExplicitCopy(N, blockSize, 
			             numBlocks, "explicitCopy")
    );

    testConfigs.push_back(
		    new PinnedMemory(N, blockSize, 
			             numBlocks, "pinnedMem")
    );
  }

  int t = 0;
  for (auto testConfig : testConfigs) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (int j = 0; j < numTests; j++) {
      std::cout << "(" << ++t << "/" << total << ")" << " ";
      auto test = testConfig->Clone();
      test->Run(&outfile);
      hipDeviceSynchronize();
      delete test;
    }
    auto finish = high_resolution_clock::now();
    auto dur = duration<double, std::milli>(finish - start);
    outfile << testConfig->name << "," << testConfig->N << "," 
	    << -1 << "," << -1 << "," << -1 << ","
	    << dur.count() << "," << std::endl
    ;
    delete testConfig;
  }
  return 0;
}
