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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_

#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/statusor.h"

namespace xla {
namespace spmd {

struct AutoShardingSolverRequest {
  int64_t num_nodes = 0;
  int64_t memory_budget = -1;
  std::vector<int> s_len;
  std::vector<int> s_follow;
  std::vector<std::pair<int, int>> e;
  std::vector<std::vector<int>> live;
  std::vector<std::vector<double>> c;
  std::vector<std::vector<double>> d;
  std::vector<std::vector<double>> m;
  std::vector<std::vector<double>> r;
  std::vector<std::pair<int, int>> a;
  std::vector<std::vector<double>> v;
  std::vector<std::string> instruction_names;
  std::optional<int64_t> solver_timeout_in_seconds;
  bool crash_at_infinity_costs_check = false;
};

struct AutoShardingSolverResult {
 public:
  AutoShardingSolverResult(
      StatusOr<std::tuple<std::vector<int64_t>, std::vector<int64_t>, double>>
          status,
      bool skip_auto_sharding)
      : status(status), skip_auto_sharding(skip_auto_sharding) {}
  StatusOr<std::tuple<std::vector<int64_t>, std::vector<int64_t>, double>>
      status;
  bool skip_auto_sharding;
};

AutoShardingSolverResult CallORToolsSolver(
    const AutoShardingSolverRequest& request);

enum AutoShardingViolationCode {
  kAliasViolationCode,     // Some node's strategy does not match its alias
  kFollowerViolationCode,  // Some node's strategy does not match its follower
  kMemoryViolationCode,    // The solution eclipses the memory budget
};

// Captures the metrics and constraint violations for the sharding result.
struct AutoShardingEvaluation {
  // A set of constraint violations; should be empty for any viable solution.
  absl::flat_hash_set<AutoShardingViolationCode> violation_codes;

  // A breakdown of each individual cost component.
  double total_communication_cost = 0.0;
  double total_computation_cost = 0.0;
  double total_resharding_cost = 0.0;

  // The total (global) objective cost.
  double total_cost = 0.0;

  bool operator==(const AutoShardingEvaluation& other) const;
};

// Evaluates the given solver result w.r.t. the input request, computing various
// solution quality metrics and validating the consistency of hard constraints.
AutoShardingEvaluation Evaluate(const AutoShardingSolverRequest& request,
                                const AutoShardingSolverResult& result);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_SOLVER_H_
