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
#define EIGEN_USE_THREADS

#include "xla/service/hlo_runner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/layout_util.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/logging.h"

namespace xla {

HloRunner::HloRunner(se::Platform* platform, int intra_op_parallelism_threads) {
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  backend_options.set_intra_op_parallelism_threads(
      intra_op_parallelism_threads);
  backend_ = Backend::CreateBackend(backend_options).value();
  device_shape_representation_fn_ = [this](const Shape& shape) {
    return backend_->compiler()->DefaultDeviceShapeRepresentation(shape);
  };
  VLOG(1) << "Created HloRunner for platform: " << platform->Name();
}

HloRunner::~HloRunner() {}

StatusOr<ScopedShapedBuffer> HloRunner::TransferLiteralToDevice(
    const Literal& literal, int64_t param_no) {
  auto shape_representation_fn = [this, param_no](const Shape& shape) {
    Shape new_shape = device_shape_representation_fn_(shape);
    if (entry_computation_layout_ == nullptr) {
      return new_shape;
    }

    Shape entry_computation_shape =
        entry_computation_layout_->parameter_shape(param_no);
    // Favor entry computation shape with some adjustment.
    ShapeUtil::ForEachMutableSubshape(
        &new_shape,
        [&entry_computation_shape](Shape* subshape, const ShapeIndex& index) {
          if (!subshape->IsArray()) {
            return;
          }
          Shape entry_computation_subshape =
              ShapeUtil::GetSubshape(entry_computation_shape, index);
          if (entry_computation_subshape.is_static() &&
              !entry_computation_subshape.layout().tiles().empty() &&
              *subshape != entry_computation_subshape) {
            *subshape = entry_computation_subshape;
          }
        });
    return new_shape;
  };

  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer buffer,
      backend().transfer_manager()->AllocateScopedShapedBuffer(
          literal.shape(), backend().memory_allocator(),
          backend().default_device_ordinal(), shape_representation_fn));
  TF_ASSIGN_OR_RETURN(
      auto stream, backend().BorrowStream(backend().default_stream_executor()));
  TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
      stream.get(), literal, buffer));
  return std::move(buffer);
}

StatusOr<std::vector<ScopedShapedBuffer>> HloRunner::TransferLiteralsToDevice(
    absl::Span<const Literal* const> literals) {
  std::vector<ScopedShapedBuffer> buffers;
  buffers.reserve(literals.size());
  for (auto i = 0; i < literals.size(); i++) {
    const Literal* literal = literals[i];
    CHECK(literal != nullptr);
    TF_ASSIGN_OR_RETURN(ScopedShapedBuffer buffer,
                        TransferLiteralToDevice(*literal, i));
    buffers.push_back(std::move(buffer));
  }
  return std::move(buffers);
}

StatusOr<std::vector<ScopedShapedBuffer>> HloRunner::TransferLiteralsToDevice(
    absl::Span<const Literal> literals) {
  std::vector<const Literal*> literal_pointers;
  literal_pointers.reserve(literals.size());
  for (const auto& literal : literals) {
    literal_pointers.push_back(&literal);
  }
  return TransferLiteralsToDevice(literal_pointers);
}

StatusOr<Literal> HloRunner::TransferLiteralFromDevice(
    const ShapedBuffer& buffer) {
  TF_ASSIGN_OR_RETURN(
      auto stream, backend().BorrowStream(backend().default_stream_executor()));

  if (buffer.on_device_shape().is_static()) {
    return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                   buffer);
  }

  Shape device_shape = buffer.on_device_shape();
  // Read real literal's shape first.
  TF_RETURN_IF_ERROR(backend().transfer_manager()->ReadDynamicShapes(
      stream.get(), &buffer, &device_shape));

  ShapedBuffer shaped_buffer(device_shape, buffer.device_ordinal());
  // Populate buffer element by element since the shapes differ now.
  shaped_buffer.buffers().ForEachMutableElement(
      [&](const xla::ShapeIndex& index, se::DeviceMemoryBase* base_buffer) {
        *base_buffer = buffer.buffer(index);
      });
  return backend().transfer_manager()->TransferLiteralFromDevice(stream.get(),
                                                                 shaped_buffer);
}

StatusOr<Literal> HloRunner::Execute(std::unique_ptr<HloModule> module,
                                     absl::Span<const Literal* const> arguments,
                                     bool run_hlo_passes,
                                     ExecutionProfile* profile) {
  xla::UpdateEntryComputationLayout(module.get(),
                                    device_shape_representation_fn_);
  entry_computation_layout_ = &(module->entry_computation_layout());
  TF_ASSIGN_OR_RETURN(std::vector<ScopedShapedBuffer> argument_buffers,
                      TransferLiteralsToDevice(arguments));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithMovedDeviceBuffers(
                          /*module=*/std::move(module),
                          /*arguments=*/std::move(argument_buffers),
                          /*run_hlo_passes=*/run_hlo_passes,
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

StatusOr<Literal> HloRunner::ExecuteWithExecutable(
    Executable* executable, absl::Span<const Literal* const> arguments,
    ExecutionProfile* profile) {
  entry_computation_layout_ =
      &(executable->module().entry_computation_layout());
  TF_ASSIGN_OR_RETURN(std::vector<ScopedShapedBuffer> argument_buffers,
                      TransferLiteralsToDevice(arguments));
  TF_ASSIGN_OR_RETURN(ExecutionOutput result,
                      ExecuteWithDeviceBuffers(
                          /*executable=*/executable,
                          /*arguments=*/argument_buffers,
                          /*profile=*/profile));
  return TransferLiteralFromDevice(result.Result());
}

// Create a partially owning vector of `ExecutionInput`s based on an owning
// vector of `OwningDeviceMemory`'s.
//
// This function creates owning references to memory which is already
// owned by a ScopedShapedBuffer. This can result in double-free and similar
// problems in rare cases (for example when the running of the HLO is
// unsuccessful). We keep this here because too much code depends on it for
// repeatedly running HLOs without reallocating device buffers.
static std::vector<ExecutionInput> ExecutionInputsFromScopedShapedBuffers(
    absl::Span<ScopedShapedBuffer const> inputs,
    HloInputOutputAliasConfig alias_config, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  std::vector<ExecutionInput> execution_inputs;

  for (int param_num = 0; param_num < inputs.size(); param_num++) {
    const ScopedShapedBuffer& input_buffer = inputs[param_num];
    ShapeTree<MaybeOwningDeviceMemory> buffer_tree(
        input_buffer.on_device_shape());

    input_buffer.buffers().ForEachElement(
        [&](const ShapeIndex& index,
            const se::DeviceMemoryBase& execution_input_buffer) {
          if (alias_config.ParameterHasAlias(param_num, index)) {
            // Store owned.
            *buffer_tree.mutable_element(index) = se::OwningDeviceMemory{
                execution_input_buffer, device_ordinal, allocator};
          } else {
            // Store unowned.
            *buffer_tree.mutable_element(index) = execution_input_buffer;
          }
        });
    execution_inputs.emplace_back(std::move(buffer_tree));
  }
  return execution_inputs;
}

// Convert the owning buffer of inputs into a (partially) owning vector of
// ExecutionInputs, and an owning vector of `OwningDeviceMemory`'s.
static void ExecutionInputsFromMovedScopedShapedBuffers(
    std::vector<ExecutionInput>* out_execution_inputs,
    std::vector<se::OwningDeviceMemory>* out_owned_args,
    std::vector<ScopedShapedBuffer> inputs,
    HloInputOutputAliasConfig alias_config, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  CHECK(out_execution_inputs->empty());
  CHECK(out_owned_args->empty());

  for (int param_num = 0; param_num < inputs.size(); param_num++) {
    ShapedBuffer input_buffer = inputs[param_num].release();

    ShapeTree<MaybeOwningDeviceMemory> buffer_tree(
        input_buffer.on_device_shape());

    input_buffer.buffers().ForEachElement(
        [&](const ShapeIndex& index,
            const se::DeviceMemoryBase& execution_input_buffer) {
          if (alias_config.ParameterHasAlias(param_num, index)) {
            VLOG(1) << "Input " << param_num << " index " << index.ToString()
                    << " buffer " << execution_input_buffer.opaque()
                    << " will be owned by out_execution_inputs.";

            // Owned by out_execution_inputs.
            // This allows the Executable to transfer the ownership to the
            // ExecutionOutput.
            *buffer_tree.mutable_element(index) = se::OwningDeviceMemory{
                execution_input_buffer, device_ordinal, allocator};
          } else {
            VLOG(1) << "Input " << param_num << " index " << index.ToString()
                    << " buffer " << execution_input_buffer.opaque()
                    << " will be owned by out_owned_args.";

            // Not owned by out_execution_inputs.
            *buffer_tree.mutable_element(index) = execution_input_buffer;
            // Owned by out_owned_args.
            out_owned_args->emplace_back(execution_input_buffer, device_ordinal,
                                         allocator);
          }
        });
    out_execution_inputs->emplace_back(std::move(buffer_tree));
  }
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithDeviceBuffers(
    std::unique_ptr<HloModule> module,
    absl::Span<ScopedShapedBuffer const> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      CreateExecutable(std::move(module), run_hlo_passes));
  return ExecuteWithDeviceBuffers(executable.get(), arguments, profile);
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithDeviceBuffers(
    Executable* executable, absl::Span<ScopedShapedBuffer const> arguments,
    ExecutionProfile* profile) {
  std::vector<ExecutionInput> execution_arguments =
      ExecutionInputsFromScopedShapedBuffers(
          arguments, executable->module().input_output_alias_config(),
          backend().default_stream_executor()->device_ordinal(),
          backend().default_stream_executor()->GetAllocator());
  return ExecuteWithExecutionInputs(executable, std::move(execution_arguments),
                                    profile);
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithMovedDeviceBuffers(
    std::unique_ptr<HloModule> module,
    std::vector<ScopedShapedBuffer> arguments, bool run_hlo_passes,
    ExecutionProfile* profile) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      CreateExecutable(std::move(module), run_hlo_passes));
  return ExecuteWithMovedDeviceBuffers(executable.get(), std::move(arguments),
                                       profile);
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithMovedDeviceBuffers(
    Executable* executable, std::vector<ScopedShapedBuffer> arguments,
    ExecutionProfile* profile) {
  std::vector<ExecutionInput> execution_arguments;
  // We need this to keep the arguments not owned by execution_arguments
  // alive.
  std::vector<se::OwningDeviceMemory> owned_arguments;

  ExecutionInputsFromMovedScopedShapedBuffers(
      &execution_arguments, &owned_arguments, std::move(arguments),
      executable->module().input_output_alias_config(),
      backend().default_stream_executor()->device_ordinal(),
      backend().default_stream_executor()->GetAllocator());

  TF_ASSIGN_OR_RETURN(ExecutionOutput retval,
                      ExecuteWithExecutionInputs(
                          executable, std::move(execution_arguments), profile));

  // This is here to make sure that the output buffers get freed up when the
  // ExecutionOutput is destroyed.
  retval.Commit();
  return retval;
}

StatusOr<ExecutionOutput> HloRunner::ExecuteWithExecutionInputs(
    Executable* executable, std::vector<ExecutionInput> arguments,
    ExecutionProfile* profile) {
  xla::UpdateEntryComputationLayout(&executable->module(),
                                    device_shape_representation_fn_);

  // Get service run options.
  se::Stream stream(backend().default_stream_executor());
  stream.Init();
  ServiceExecutableRunOptions service_run_options =
      GetServiceRunOptionsForDevice(backend().default_device_ordinal(), &stream,
                                    nullptr, RunId());
  service_run_options.mutable_run_options()->set_execution_profile(profile);

  TF_ASSIGN_OR_RETURN(ExecutionOutput retval,
                      executable->ExecuteOnStreamWrapper(&service_run_options,
                                                         std::move(arguments)));
  TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return std::move(retval);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module, const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      CreateExecutable(std::move(module), options.run_hlo_passes));
  return ExecuteReplicated(executable.get(), options, device_assignment);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicatedImpl(
    std::function<StatusOr<std::vector<ScopedShapedBuffer>>(
        const std::vector<ServiceExecutableRunOptions>&,
        const std::vector<absl::Span<const ShapedBuffer* const>>&)>
        execution_helper,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  std::vector<std::unique_ptr<se::Stream>> streams;
  std::vector<ServiceExecutableRunOptions> service_run_options;
  int64_t num_partitions = device_assignment->computation_count();

  std::vector<ScopedShapedBuffer> argument_buffers;
  // This reserve() call is necessary for correctness, because
  // argument_buffer_ptrs contains pointers into the elements of
  // argument_buffers.
  const int64_t total_argument_count = [&]() {
    int64_t total = 0;
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      total += argument_count_provider(i);
    }
    return total;
  }();
  argument_buffers.reserve(total_argument_count);

  // Plus one so we can safely get &argument_buffer_ptrs[0] in case there are
  // no arguments.
  std::vector<const ShapedBuffer*> argument_buffer_ptrs(total_argument_count +
                                                        1);
  std::vector<absl::Span<const ShapedBuffer* const>> argument_buffer_slices;
  int64_t index = 0;
  RunId run_id;
  for (int64_t i = 0; i < options.num_replicas; ++i) {
    int64_t device =
        (*device_assignment)(i / num_partitions, i % num_partitions);
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        backend().stream_executor(device));
    streams.push_back(std::make_unique<se::Stream>(executor));
    streams.back()->Init();
    service_run_options.emplace_back(GetServiceRunOptionsForDevice(
        device, streams.back().get(), device_assignment, run_id));

    // Copy arguments to device.
    const int64_t argument_count = argument_count_provider(i);
    for (int64_t arg_index = 0; arg_index < argument_count; arg_index++) {
      const Literal* const argument = argument_provider(i, arg_index);
      TF_RET_CHECK(argument != nullptr);
      TF_ASSIGN_OR_RETURN(
          ScopedShapedBuffer argument_buffer,
          backend().transfer_manager()->AllocateScopedShapedBuffer(
              argument->shape(), backend().memory_allocator(), device,
              device_shape_representation_fn_));
      TF_RETURN_IF_ERROR(backend().transfer_manager()->TransferLiteralToDevice(
          streams.back().get(), *argument, argument_buffer));
      argument_buffers.push_back(std::move(argument_buffer));
      argument_buffer_ptrs[index++] = &argument_buffers.back();
    }
    argument_buffer_slices.emplace_back(
        &argument_buffer_ptrs[index - argument_count], argument_count);
  }

  std::unique_ptr<tsl::thread::ThreadPool> pool;
  TF_RET_CHECK(options.infeed_values.empty() ||
               options.infeed_values.size() == options.num_replicas);
  int64_t num_threads = options.infeed_values.size();
  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    num_threads += options.num_replicas;
  }
  if (num_threads > 0) {
    pool = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "infeed_outfeed",
        /*num_threads=*/num_threads);
  }
  if (!options.infeed_values.empty()) {
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      int64_t device =
          (*device_assignment)(i / num_partitions, i % num_partitions);
      pool->Schedule([this, device, &options, i]() {
        se::StreamExecutor* executor =
            backend().stream_executor(device).value();
        VLOG(1) << "Starting infeed on device " << device;
        for (int64_t step = 1;
             options.infeed_steps < 0 || step <= options.infeed_steps; ++step) {
          TF_CHECK_OK(backend().transfer_manager()->TransferLiteralToInfeed(
              executor, *options.infeed_values[i]));
          if (step % 100 == 0) {
            VLOG(1) << "Infeed step " << step;
          }
        }
      });
    }
  }
  if (ShapeUtil::IsInitialized(options.outfeed_shape)) {
    if (options.outfeed_values) {
      options.outfeed_values->resize(options.num_replicas);
    }
    for (int64_t i = 0; i < options.num_replicas; ++i) {
      int64_t device =
          (*device_assignment)(i / num_partitions, i % num_partitions);
      pool->Schedule([this, device, &options, i]() {
        se::StreamExecutor* executor =
            backend().stream_executor(device).value();
        VLOG(1) << "Starting outfeed on device " << device;
        for (int64_t step = 1;
             options.infeed_steps < 0 || step <= options.infeed_steps; ++step) {
          Literal literal(options.outfeed_shape);
          TF_CHECK_OK(backend().transfer_manager()->TransferLiteralFromOutfeed(
              executor, &literal));
          if (options.outfeed_values) {
            options.outfeed_values->at(i) = std::move(literal);
          }
          if (step % 100 == 0) {
            VLOG(1) << "Outfeed step " << step;
          }
        }
      });
    }
  }

  VLOG(1) << "Replicated execution started";
  TF_ASSIGN_OR_RETURN(
      std::vector<ScopedShapedBuffer> results,
      execution_helper(service_run_options, argument_buffer_slices));
  VLOG(1) << "Replicated execution terminated";

  std::vector<Literal> exec_results;
  exec_results.reserve(options.num_replicas);
  for (int64_t i = 0; i < options.num_replicas; ++i) {
    TF_RETURN_IF_ERROR(streams[i]->BlockHostUntilDone());
    TF_ASSIGN_OR_RETURN(Literal literal,
                        backend().transfer_manager()->TransferLiteralFromDevice(
                            streams[i].get(), results[i]));
    exec_results.push_back(std::move(literal));
  }
  return std::move(exec_results);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    Executable* executable, const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment, ExecutionProfile* profile) {
  return ExecuteReplicatedImpl(
      [&](const std::vector<ServiceExecutableRunOptions>& service_run_options,
          const std::vector<absl::Span<const ShapedBuffer* const>>&
              argument_buffer_slices)
          -> StatusOr<std::vector<ScopedShapedBuffer>> {
        std::vector<ScopedShapedBuffer> results;
        if (!options.use_threads) {
          TF_ASSIGN_OR_RETURN(
              results, executable->ExecuteOnStreams(service_run_options,
                                                    argument_buffer_slices));
        } else {
          absl::Mutex mutex;
          std::vector<StatusOr<ScopedShapedBuffer>> thread_results(
              options.num_replicas);
          {
            VLOG(1) << "Creating thread pool for " << options.num_replicas
                    << " replicas";
            tsl::thread::ThreadPool pool(tsl::Env::Default(), "replicas",
                                         options.num_replicas);
            for (int64_t i = 0; i < options.num_replicas; ++i) {
              pool.Schedule([&, i] {
                auto result = executable->ExecuteOnStream(
                    &service_run_options[i], argument_buffer_slices[i],
                    nullptr);
                absl::MutexLock lock(&mutex);
                thread_results[i] = std::move(result);
              });
            }

            // Note: the thread pool destructor guarantees it completes all
            // work before we leave this scope.
          }
          for (auto& thread_result : thread_results) {
            if (!thread_result.ok()) {
              return thread_result.status();
            }
            results.push_back(std::move(thread_result).value());
          }
        }
        return results;
      },
      [&](int64_t replica) { return options.arguments.size(); },
      [&](int64_t replica, int64_t index) { return options.arguments[index]; },
      options, device_assignment);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::function<Executable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    const ReplicatedExecuteOptions& options,
    DeviceAssignment* device_assignment) {
  DeviceAssignment computation_device_assignment;
  if (device_assignment == nullptr) {
    TF_ASSIGN_OR_RETURN(
        computation_device_assignment,
        backend().computation_placer()->AssignDevices(options.num_replicas, 1));
    device_assignment = &computation_device_assignment;
  }
  CHECK_NE(device_assignment, nullptr);
  return ExecuteReplicatedImpl(
      [&](const std::vector<ServiceExecutableRunOptions>& service_run_options,
          const std::vector<absl::Span<const ShapedBuffer* const>>&
              argument_buffer_slices)
          -> StatusOr<std::vector<ScopedShapedBuffer>> {
        TF_RET_CHECK(options.use_threads);
        std::vector<ScopedShapedBuffer> results;
        absl::Mutex mutex;
        std::vector<StatusOr<ScopedShapedBuffer>> thread_results(
            options.num_replicas);
        {
          VLOG(1) << "Creating thread pool for " << options.num_replicas
                  << " replicas";
          tsl::thread::ThreadPool pool(tsl::Env::Default(), "replicas",
                                       options.num_replicas);
          for (int64_t i = 0; i < options.num_replicas; ++i) {
            for (const auto& arg : argument_buffer_slices[i]) {
              TF_RET_CHECK(arg != nullptr);
            }
            pool.Schedule([&, i] {
              auto result = executable_provider(i)->ExecuteOnStream(
                  &service_run_options[i], argument_buffer_slices[i], nullptr);
              absl::MutexLock lock(&mutex);
              thread_results[i] = std::move(result);
            });
          }

          // Note: the thread pool destructor guarantees it completes all work
          // before we leave this scope.
        }
        for (auto& thread_result : thread_results) {
          if (!thread_result.ok()) {
            return thread_result.status();
          }
          results.push_back(std::move(thread_result).value());
        }
        return results;
      },
      argument_count_provider, argument_provider, options, device_assignment);
}

StatusOr<std::vector<Literal>> HloRunner::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    const ReplicatedExecuteOptions& options) {
  TF_ASSIGN_OR_RETURN(
      DeviceAssignment device_assignment,
      backend().computation_placer()->AssignDevices(options.num_replicas, 1));
  return ExecuteReplicated(std::move(module), options, &device_assignment);
}

StatusOr<std::unique_ptr<Executable>> HloRunner::CreateExecutable(
    std::unique_ptr<HloModule> module, bool run_hlo_passes) {
  xla::UpdateEntryComputationLayout(module.get(),
                                    device_shape_representation_fn_);
  if (run_hlo_passes) {
    auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
    TF_ASSIGN_OR_RETURN(
        auto executables,
        backend().compiler()->Compile(std::move(module_group),
                                      {{backend().default_stream_executor()}},
                                      backend().memory_allocator()));
    return std::move(executables[0]);
  }
  return backend().compiler()->RunBackend(std::move(module),
                                          backend().default_stream_executor(),
                                          backend().memory_allocator());
}

ServiceExecutableRunOptions HloRunner::GetServiceRunOptionsForDevice(
    int64_t device, se::Stream* stream, DeviceAssignment* device_assignment,
    RunId run_id) {
  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(device);
  run_options.set_stream(stream);
  run_options.set_allocator(backend().memory_allocator());
  run_options.set_intra_op_thread_pool(
      backend().eigen_intra_op_thread_pool_device());
  if (device_assignment != nullptr) {
    run_options.set_device_assignment(device_assignment);
  }
  run_options.set_run_id(run_id);
  return ServiceExecutableRunOptions(run_options,
                                     backend().StreamBorrowerWithPriority());
}

Backend& HloRunner::backend() {
  if (!backend_) {
    backend_ = Backend::CreateDefaultBackend().value();
    VLOG(1) << "Executing on platform " << backend().platform()->Name();
  }
  return *backend_;
}

const Backend& HloRunner::backend() const {
  return const_cast<HloRunner*>(this)->backend();
}

absl::string_view HloRunner::Name() const {
  return backend_->platform()->Name();
}

}  // namespace xla
