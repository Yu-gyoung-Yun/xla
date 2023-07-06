/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/gpu_device_info.h" // [yg]
// #include "xla/service/gpu/gpu_executable.h" // [yg]
#include "xla/stream_executor/gpu/gpu_executor.h" // [yg]
#include "xla/service/gpu/gpu_compiler.h" //[yg]
#include "xla/service/gpu/gpu_executable.h" // [yg]
#include "xla/tests/hlo_test_base.h"
#include <nvtx3/nvToolsExt.h> // [yg]
#include "xla/tests/client_library_test_base.h" // [yg]
#include "xla/client/client_library.h" // [yg]
namespace xla {
namespace gpu {

LocalClient* GetOrCreateLocalClientOrDie(
    const LocalClientOptions& client_options) {
  StatusOr<LocalClient*> result =
      ClientLibrary::GetOrCreateLocalClient(client_options);
  TF_CHECK_OK(result.status()) << " could not create local client for testing";
  return result.value();
}

namespace se = stream_executor; // [yg]

class GpuHloCostAnalysisTest : public HloTestBase {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  HloCostAnalysis::Options options_{ShapeSizeBytesFunction(),
                                    /*per_second_rates=*/{},
                                    /*count_multiple_input_accesses=*/true};
  GpuHloCostAnalysis analysis_{options_};
  GpuHloCostAnalysisTest() : HloTestBase() {}
};

TEST_F(GpuHloCostAnalysisTest, ConvCustomCall) {
  absl::string_view hlo_string = R"(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = s8[128,12,24,24,4]{4,3,2,1,0} parameter(0)
  p1 = s8[16,12,5,5,4]{4,3,2,1,0} parameter(1)
  p2 = f32[16]{0} parameter(2)
  conv1 = (s8[128,4,24,24,4]{4,3,2,1,0}, u8[0]{0}) custom-call(p0, p1, p2),
              window={size=5x5 pad=2_2x2_2},
              dim_labels=bf01_oi01->bf01,
              custom_call_target="__cudnn$convBiasActivationForward"
  ROOT tuple = tuple(conv1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string)); // unique_ptr<xla::VerifiedHloModule>
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));
  HloComputation* comp = module->entry_computation();
  const HloInstruction* conv1 = comp->GetInstructionWithName("conv1");
  int op0_size = sizeof(int8_t) * 128 * 12 * 24 * 24 * 4;
  int op1_size = sizeof(int8_t) * 16 * 12 * 5 * 5 * 4;
  int op2_size = sizeof(float) * 16;
  int out_size = sizeof(int8_t) * 128 * 4 * 24 * 24 * 4;
  std::cout<<"analysis_.operand_bytes_accessed(*conv1, 0): "<<analysis_.operand_bytes_accessed(*conv1, 0)<<std::endl;
  // operand_num: instruction.operand_count()
  //float total_bytes_accessed = cost_analysis_.bytes_accessed(instruction);
  //std::cout<<"analysis_.bytes_accessed(instruction): "<<analysis_.bytes_accessed(&conv1)<<std::end;
  EXPECT_EQ(analysis_.operand_bytes_accessed(*conv1, /*operand.first(operand_num)=*/0 /*operand.second(index)=*/), op0_size);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*conv1, 1), op1_size);
  EXPECT_EQ(analysis_.operand_bytes_accessed(*conv1, 2), op2_size);
  EXPECT_EQ(analysis_.output_bytes_accessed(*conv1), out_size);
  EXPECT_EQ(analysis_.bytes_accessed(*conv1),
            op0_size + op1_size + op2_size + out_size);
  EXPECT_EQ(analysis_.flop_count(*conv1), 159694848);

  std::string test_platform = "cuda";
  #if TENSORFLOW_USE_ROCM
    test_platform = "rocm";
  #endif
  // GPU 0
  /*se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(test_platform).value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  const xla::gpu::GpuDeviceInfo dev_info = xla::gpu::GetGpuDeviceInfo(executor); //from gpu_device_info.cc // GetGpuDeviceInfo(stream_exec->GetDeviceDescription());
  absl::string_view name(dev_info.name);
  std::cout<<"gpu NAME: "<<name<<"\n"; // NVIDIA A100-SXM4-80GB
  std::cout<<"gpu threads_per_block_limit: "<<dev_info.threads_per_block_limit<<"\n"; //1024
  
  // GPU 1
  se::StreamExecutor* executor_ = platform->ExecutorForDevice(1).value();
  const xla::gpu::GpuDeviceInfo dev_info_ = xla::gpu::GetGpuDeviceInfo(executor);
  absl::string_view name_(dev_info_.name);
  std::cout<<"gpu NAME: "<<name_<<"\n"; // NVIDIA A100-SXM4-80GB
  std::cout<<"gpu threads_per_block_limit: "<<dev_info_.threads_per_block_limit<<"\n"; //1024

  se::gpu::GpuExecutor* gpu_executor = se::gpu::ExtractGpuExecutor(executor);
  
  // Build Executable // != compiled_module (이거는 RunHloPass로 해야해)
  auto a = GpuCompiler(); */
  // GpuCompiler::RunBackend(/*std::unique_ptr<HloModule>= */std::move(module), /*se::StreamExecutor*= */executor, /*const Compiler::CompileOptions& =*/{/*device_allocator=*/nullptr});
  // se::gpu::ScopedActivateExecutorContext activation(gpu_executor);

  //  ref::gpu_compiler_test.cc
  std::unique_ptr<xla::Executable> gpu_executable =
      backend()
          .compiler()
          ->RunBackend(module->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();// RunBackend로 하면 unique_ptr<xla::Executable>
  //EXPECT_EQ(OkStatus(), backend()
  //                          .compiler()
  //                          ->RunBackend(std::move(module),
  //                                       backend().default_stream_executor(),
  //                                       /*device_allocator=*/nullptr)
  //                          .status());
  
  // gpu_compiler_test.cc
  const char* hlo_text = R"(
  HloModule cluster

  ENTRY main {
    cst = f32[1]{0} constant({0})
    ROOT tuple_out = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}) tuple(cst, cst, cst, cst)
  }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0, 0}));
  std::cout<<"After RunAndCompare: \n";
  std::cout<<"backend().compiler(): "<<backend().compiler()<<"\n";
  TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        backend().compiler()->RunBackend(
            std::move(module), backend().default_stream_executor(),
            backend().default_stream_executor()->GetAllocator()));
  GpuExecutable* g_executable =
      static_cast<GpuExecutable*>(executable.get()); //  executable.get() == params?
  absl::Span<const BufferAllocation> allocations =
      g_executable->GetAllocations();
  std::cout<<"BufferAllocation Size: "<<allocations.size()<<std::endl;
  

  //std::cout<<"gpu_compiler\n";
  //xla::gpu::GpuCompiler gpu_compiler;
  //gpu_runtime_executable_
  //std::unique_ptr<GpuRuntimeExecutable> gpu_runtime = g_executable->gpu_runtime_executable_;
  
  //auto param = &g_executable->Params; //  g_executable->Params;
  //GpuExecutable::Params param;
  //std::unique_ptr<GpuExecutable> a = g_executable->Create(param);

  // Check GPU Device
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName(test_platform).value();
  se::StreamExecutor* gpu_executor = platform->ExecutorForDevice(0).value();
  const auto& description = gpu_executor->GetDeviceDescription();
  se::CudaComputeCapability cc = description.cuda_compute_capability(); // 8.0 -> A100
  std::cout<<"cc.ToString(): "<<cc.ToString()<<std::endl;
  //  ref:: platform_tuil.cc
  /*if (gpu_executor->platform()->id() == se::cuda::kCudaPlatformId) {
    // CUDA devices must have a minimum compute capability.
    se::CudaComputeCapability cc = description.cuda_compute_capability();
    if (!cc.IsAtLeast(kMinCudaComputeCapabilityMajor,
                      kMinCudaComputeCapabilityMinor)) {
      LOG(INFO) << "StreamExecutor cuda device (" << gpu_executor->device_ordinal()
                << ") is of "
                << "insufficient compute capability: "
                << kMinCudaComputeCapabilityMajor << "."
                << kMinCudaComputeCapabilityMinor << " required, "
                << "device is " << cc.ToString();
      return false;
  }*/

  //  ref:  client_library_test_base.cc
  
}

}  // namespace gpu
}  // namespace xla


// ref: layout_assignment_test.cc
/*EXPECT_EQ(OkStatus(), backend()
                            .compiler()
                            ->RunBackend(std::move(compiled_module),
                                         backend().default_stream_executor(),
                                         /*device_allocator=*/ //nullptr)
                            //.status());*/
