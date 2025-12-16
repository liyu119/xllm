/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "kernel_handle.h"

namespace xllm::kernel::npu {

class KernelLoader {
 public:
  static KernelLoader& get_instance();

  KernelHandle load_kernel(const std::string& kernel_name,
                           const std::string& binary_path);

  std::vector<KernelHandle> load_kernels(
      const std::vector<std::pair<std::string, std::string>>& kernel_configs);

  int load_kernels_from_directory(const std::string& directory_path);

  KernelHandle get_kernel(const std::string& kernel_name);

  bool get_kernel_workspace_config(const std::string& kernel_name,
                                   int64_t& workspace_size,
                                   int64_t& lock_init_value,
                                   int64_t& lock_num);

  void cleanup();

 private:
  KernelLoader() = default;
  ~KernelLoader() = default;

  KernelLoader(const KernelLoader&) = delete;
  KernelLoader& operator=(const KernelLoader&) = delete;
};

}  // namespace xllm::kernel::npu


