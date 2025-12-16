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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm::kernel::npu {

class KernelHandle;

class KernelRegistry {
 public:
  static KernelRegistry& get_instance();

  KernelRegistry(const KernelRegistry&) = delete;
  KernelRegistry& operator=(const KernelRegistry&) = delete;

  bool register_kernel(const std::string& kernel_name,
                       const std::string& binary_path);

  bool register_kernels(
      const std::vector<std::pair<std::string, std::string>>& kernel_configs);

  KernelHandle get_kernel_handle(const std::string& kernel_name) const;

  bool is_kernel_registered(const std::string& kernel_name) const;

  std::vector<std::string> get_registered_kernels() const;

  void cleanup();

  bool parse_json_config(const std::string& json_path,
                         std::string& kernel_name,
                         std::string& mix_mode,
                         int64_t& workspace_size,
                         int64_t& lock_init_value,
                         int64_t& lock_num);

  bool get_kernel_workspace_config(const std::string& kernel_name,
                                   int64_t& workspace_size,
                                   int64_t& lock_init_value,
                                   int64_t& lock_num) const;

 private:
  KernelRegistry() = default;
  ~KernelRegistry();

  struct KernelInfo {
    std::string name;
    char* buffer;
    void* stub_func;
    std::string persistent_func_name;
    std::string mix_mode;
    int64_t workspace_size;
    int64_t lock_init_value;
    int64_t lock_num;
  };

  char* load_binary_file(const std::string& file_path, uint32_t& file_size);
  bool register_binary(KernelInfo& info,
                       uint32_t binary_size,
                       void** stub_func);

  std::unordered_map<std::string, KernelInfo> kernel_infos_;
};

}  // namespace xllm::kernel::npu


