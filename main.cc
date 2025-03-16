#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <chrono>

constexpr const unsigned int arrayLength = 1 << 24;
constexpr const unsigned int bufferSize = arrayLength * sizeof(float);

constexpr const char* shaderSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}
)";

class MetalAdder {
 public:
  MetalAdder(MTL::Device* device);
  ~MetalAdder();
  void prepareData();
  void sendComputeCommand();

 private:
  void generateRandomFloatData(MTL::Buffer* buffer);
  void encodeAddCommand(MTL::ComputeCommandEncoder* computeEncoder);
  void verifyResults();

  MTL::Device* _mDevice;
  MTL::ComputePipelineState* _mAddFunctionPSO;
  MTL::CommandQueue* _mCommandQueue;
  MTL::Buffer* _mBufferA;
  MTL::Buffer* _mBufferB;
  MTL::Buffer* _mBufferResult;
};

MetalAdder::MetalAdder(MTL::Device* device) : _mDevice(device->retain()) {
  NS::Error* error = nullptr;
  MTL::Library* defaultLibrary = _mDevice->newLibrary(
      NS::String::string(shaderSrc, NS::UTF8StringEncoding), nullptr, &error);
  assert(defaultLibrary != nullptr);

  MTL::Function* addFunction = defaultLibrary->newFunction(
      NS::String::string("add_arrays", NS::UTF8StringEncoding));
  assert(addFunction != nullptr);

  _mAddFunctionPSO = _mDevice->newComputePipelineState(addFunction, &error);
  assert(_mAddFunctionPSO != nullptr);

  _mCommandQueue = _mDevice->newCommandQueue();
  assert(_mCommandQueue != nullptr);

  addFunction->release();
  defaultLibrary->release();
}

MetalAdder::~MetalAdder() {
  _mCommandQueue->release();
  _mAddFunctionPSO->release();
  _mDevice->release();
}

void MetalAdder::prepareData() {
  _mBufferA = _mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  _mBufferB = _mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  _mBufferResult =
      _mDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

  generateRandomFloatData(_mBufferA);
  generateRandomFloatData(_mBufferB);
}

void MetalAdder::sendComputeCommand() {
  MTL::CommandBuffer* commandBuffer = _mCommandQueue->commandBuffer();
  assert(commandBuffer != nullptr);

  MTL::ComputeCommandEncoder* computeEncoder =
      commandBuffer->computeCommandEncoder();
  assert(computeEncoder != nullptr);
  encodeAddCommand(computeEncoder);
  computeEncoder->endEncoding();

  auto a = std::chrono::system_clock::now();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  auto b = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  printf("%lld ms\n", elapsed);
  printf("BW: %.3f GB/s\n",
         2000.0f * bufferSize / 1024 / 1024 / 1024 / elapsed);
  verifyResults();
}

void MetalAdder::generateRandomFloatData(MTL::Buffer* buffer) {
  float* dataPtr = (float*)buffer->contents();
  for (unsigned long index = 0; index < arrayLength; ++index) {
    dataPtr[index] = (float)rand() / (float)(RAND_MAX);
  }
}

void MetalAdder::encodeAddCommand(MTL::ComputeCommandEncoder* computeEncoder) {
  computeEncoder->setComputePipelineState(_mAddFunctionPSO);
  computeEncoder->setBuffer(_mBufferA, 0, 0);
  computeEncoder->setBuffer(_mBufferB, 0, 1);
  computeEncoder->setBuffer(_mBufferResult, 0, 2);

  MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

  NS::UInteger threadGroupSize =
      _mAddFunctionPSO->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > arrayLength) {
    threadGroupSize = arrayLength;
  }
  MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadgroupSize);
}

void MetalAdder::verifyResults() {
  float* a = (float*)_mBufferA->contents();
  float* b = (float*)_mBufferB->contents();
  float* result = (float*)_mBufferResult->contents();

  for (unsigned long index = 0; index < arrayLength; ++index) {
    if (result[index] != (a[index] + b[index])) {
      printf("Compute ERROR: index=%lu result=%g vs %g=a+b\n", index,
             result[index], a[index] + b[index]);
      assert(result[index] == (a[index] + b[index]));
    }
  }
}

int main() {
  NS::AutoreleasePool* pAutoreleasePool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* device = MTL::CreateSystemDefaultDevice();
  {
    MetalAdder adder(device);
    adder.prepareData();
    adder.sendComputeCommand();
  }
  pAutoreleasePool->release();
  return 0;
}
