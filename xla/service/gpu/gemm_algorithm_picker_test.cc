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

#include "xla/service/gpu/gemm_algorithm_picker.h"

#include <string>

#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/protobuf/dnn.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GemmAlgorithmPickerTest : public HloTestBase {
 public:
  GemmAlgorithmPickerTest() { AutotunerUtil::ClearAutotuneResults(); }
};

TEST_F(GemmAlgorithmPickerTest, SetAlgorithm) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  ROOT %dot = f32[100,100]{1,0} dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];
  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .cuda_compute_capability()),
                          m.get()));
  changed = false;
  DebugOptions opts;
  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_algo_id = result.algorithm().algo_id();
  int64_t new_algo_id = old_algo_id + 1;
  result.mutable_gemm()->set_algorithm(new_algo_id);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GemmAlgorithmPicker again.  The dot should
  // have the new algorithm.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .cuda_compute_capability()),
                          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  SCOPED_TRACE(m->ToString());
  HloInstruction* dot;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(&dot)));

  TF_ASSERT_OK_AND_ASSIGN(GemmBackendConfig config,
                          dot->backend_config<GemmBackendConfig>());
  EXPECT_EQ(config.selected_algorithm(), new_algo_id);
}

TEST_F(GemmAlgorithmPickerTest, GetAlgorithmWithoutDevice) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[100,100]{1,0} parameter(0)
  %arg1 = f32[100,100]{1,0} parameter(1)
  ROOT %dot = f32[100,100]{1,0} dot(arg0, arg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .cuda_compute_capability()),
                          m.get()));
  changed = false;

  DebugOptions opts;
  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};

  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GemmAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_algo_id = result.algorithm().algo_id();
  int64_t new_algo_id = old_algo_id + 1;
  result.mutable_gemm()->set_algorithm(new_algo_id);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GemmAlgorithmPicker again.  The dot should
  // have the new algorithm.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  changed = false;

  DevicelessConfig deviceless_config{
      stream_exec->GetDeviceDescription().model_str(),
      stream_exec->GetDeviceDescription().cuda_compute_capability()};
  AutotuneConfig deviceless_cfg{deviceless_config, opts};
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmRewriter(stream_exec->GetDeviceDescription()
                                           .cuda_compute_capability()),
                          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GemmAlgorithmPicker(deviceless_cfg), m.get()))
  ASSERT_TRUE(changed);

  SCOPED_TRACE(m->ToString());
  HloInstruction* dot;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::CustomCall(&dot)));

  TF_ASSERT_OK_AND_ASSIGN(GemmBackendConfig config,
                          dot->backend_config<GemmBackendConfig>());
  EXPECT_EQ(config.selected_algorithm(), new_algo_id);
}

}  // namespace
}  // namespace xla::gpu
