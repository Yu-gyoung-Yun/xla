/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/call_inliner.h"

#include <deque>
#include <memory>

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_domain_isolator.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

// Traverses the callee computation, inlining cloned nodes into the caller
// computation and connecting them to producers/consumers appropriately.
// When the traversal has completed, the provided call instruction is entirely
// replaced in the caller's graph.
class SubcomputationInsertionVisitor : public DfsHloVisitorWithDefault {
 public:
  // call is the call operation -- it will be replaced with the body of the
  // called computation.
  explicit SubcomputationInsertionVisitor(HloInstruction* call)
      : call_(call), outer_(call->parent()) {
    CHECK_EQ(HloOpcode::kCall, call_->opcode());
  }

  // Resolves the operands to the HLO instruction in the inlined (caller) graph,
  // and clones the HLO instruction into that graph with the new operands.
  Status DefaultAction(HloInstruction* hlo) override {
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : hlo->operands()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand, Resolve(operand));
      new_operands.push_back(new_operand);
    }
    VLOG(1) << "Cloning HLO and adding to caller: " << hlo->ToString();
    auto new_hlo = hlo->CloneWithNewOperands(hlo->shape(), new_operands);
    HloInstruction* new_hlo_pointer =
        outer_->AddInstruction(std::move(new_hlo));
    TF_RETURN_IF_ERROR(NoteMapping(hlo, new_hlo_pointer));

    // Account for control edges.
    for (HloInstruction* control_predecessor : hlo->control_predecessors()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_control_predecessor,
                          Resolve(control_predecessor));
      TF_RETURN_IF_ERROR(
          new_control_predecessor->AddControlDependencyTo(new_hlo_pointer));
    }

    return OkStatus();
  }

  // Does not create new nodes for the parameter; rather, notes the mapping from
  // the subcomputation parameter node to the call operands in the caller
  // computation.
  Status HandleParameter(HloInstruction* parameter) override {
    TF_RETURN_IF_ERROR(NoteMapping(
        parameter, call_->mutable_operand(parameter->parameter_number())));
    return OkStatus();
  }

  // Wires the consumers of the call to instead point at the newly created root,
  // replacing the call operation in the caller computation.
  Status FinishVisit(HloInstruction* root) override {
    TF_ASSIGN_OR_RETURN(HloInstruction * new_root, Resolve(root));
    VLOG(1) << "Replacing all uses of " << call_->ToString()
            << " with new root " << new_root->ToString();
    return outer_->ReplaceInstruction(call_, new_root);
  }

  CallInliner::InlinedInstructionMap ConsumeInstructionMap() {
    return std::move(subcomputation_hlo_to_new_hlo_);
  }

 private:
  // Resolves the callee subcomputation_hlo to the new (inline) HLO in the
  // caller computation, or returns a NotFound error if that subcomputation HLO
  // has not been mapped.
  StatusOr<HloInstruction*> Resolve(HloInstruction* subcomputation_hlo) {
    auto it = subcomputation_hlo_to_new_hlo_.find(subcomputation_hlo);
    if (it == subcomputation_hlo_to_new_hlo_.end()) {
      return NotFound(
          "Could not find mapping from subcomputation HLO %s to a cloned HLO.",
          subcomputation_hlo->ToString());
    }
    return it->second;
  }

  // Notes that the given subcomputation_hlo in the callee has been mapped to
  // the (inline) new_hlo in the caller computation.
  //
  // Returns an error status if the subcomputation_hlo is mapped more than
  // once.
  Status NoteMapping(HloInstruction* subcomputation_hlo,
                     HloInstruction* new_hlo) {
    auto result = subcomputation_hlo_to_new_hlo_.insert(
        std::make_pair(subcomputation_hlo, new_hlo));
    TF_RET_CHECK(result.second)
        << "A mapping for the subcomputation HLO is already present.";
    return OkStatus();
  }

  HloInstruction* call_;
  HloComputation* outer_;
  CallInliner::InlinedInstructionMap subcomputation_hlo_to_new_hlo_;
};

}  // namespace

/* static */ StatusOr<CallInliner::InlinedInstructionMap> CallInliner::Inline(
    HloInstruction* call) {
  TF_RET_CHECK(call->opcode() == HloOpcode::kCall)
      << "Instruction was not a call op: " << call->opcode();
  const auto& callees = call->called_computations();
  TF_RET_CHECK(callees.size() == 1);
  HloComputation* callee = callees[0];
  // We visit the callee, cloning its body into its caller.
  SubcomputationInsertionVisitor visitor(call);
  TF_RETURN_IF_ERROR(callee->Accept(&visitor));
  return visitor.ConsumeInstructionMap();
}

StatusOr<bool> CallInliner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  // Because call graph nodes are visited in post-order (callees before callers)
  // we'll always inline kCalls into their callers in the appropriate order.
  bool did_mutate = false;
  TF_RETURN_IF_ERROR(call_graph->VisitNodes([&](const CallGraphNode& node)
                                                -> Status {
    if (!HloInstruction::IsThreadIncluded(
            node.computation()->execution_thread(), execution_threads)) {
      return OkStatus();
    }
    VLOG(1) << "Visiting node: " << node.ToString();
    for (HloInstruction* instruction :
         node.computation()->MakeInstructionPostOrder()) {
      // Don't inline async called computation since currently it's only
      // used for parallel device computation.
      // TODO(b/229887502): update the inliner to ignore only parallel
      // device type async call instead of all.
      if (instruction->opcode() == HloOpcode::kCall &&
          !instruction->parent()->IsAsyncComputation()) {
        const auto& callees = instruction->called_computations();
        TF_RET_CHECK(callees.size() == 1);
        if (!single_call_site_ || call_graph->GetNode(instruction->to_apply())
                                          .caller_callsites()
                                          .size() == 1) {
          TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap inline_map,
                              Inline(instruction));
          if (update_domain_) {
            HloDomainIsolator isolator(
                []() { return ShardingDomainCreator{}; });
            for (const auto& [call_inst, inlined_inst] : inline_map) {
              TF_RETURN_IF_ERROR(isolator.UpdateDomains(inlined_inst).status());
            }
          }
          did_mutate = true;
        }
      }
    }
    return OkStatus();
  }));
  if (did_mutate) {
    // Run DCE to remove called computations which are now becoming unused.
    // This can result then in problems if within the called computation, there
    // were send/recv instructions, which the module group verifier will flag as
    // error findingthe same channel ID used for multiple send/recv
    // instructions.
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
  }
  return did_mutate;
}

}  // namespace xla
