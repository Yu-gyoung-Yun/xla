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

#ifndef XLA_PYTHON_IFRT_DEVICE_H_
#define XLA_PYTHON_IFRT_DEVICE_H_

#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {
namespace ifrt {

// Short-term alias to reuse `xla::PjRtDevice` without a separate abstract type.
using Device = ::xla::PjRtDevice;

// Ordered list of devices.
class DeviceList {
 public:
  using value_type = Device*;

  // Number of devices to inline in `Devices`.
  static constexpr int kInlineDeviceSize = 1;

  // TODO(hyeontaek): Consider using variant<Device*, std::vector<Device*>> for
  // better performance.
  using Devices = absl::InlinedVector<Device*, kInlineDeviceSize>;

  explicit DeviceList(Devices devices) : devices_(std::move(devices)) {}

  absl::Span<Device* const> devices() const { return devices_; }

  int size() const { return devices_.size(); }
  bool empty() const { return devices_.empty(); }

  Device* operator[](int i) const { return devices_[i]; }
  Device* at(int i) const { return devices_.at(i); }
  Device* front() const { return devices_.front(); }
  Device* back() const { return devices_.back(); }

  auto begin() const { return devices_.begin(); }
  auto cbegin() const { return devices_.cbegin(); }
  auto end() const { return devices_.end(); }
  auto cend() const { return devices_.cend(); }

 private:
  Devices devices_;
};

// Returns the id of each device in `device_list`.
std::vector<int> GetDeviceIds(DeviceList device_list);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_DEVICE_H_
