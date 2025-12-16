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

#include "kernel_loader.h"

#include <filesystem>
#include <fstream>

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "kernel_registry.h"

namespace xllm::kernel::npu {

KernelLoader& KernelLoader::get_instance() {
  static KernelLoader loader;
  return loader;
}

KernelHandle KernelLoader::load_kernel(const std::string& kernel_name,
                                       const std::string& binary_path) {
  auto& registry = KernelRegistry::get_instance();
  if (registry.register_kernel(kernel_name, binary_path)) {
    return registry.get_kernel_handle(kernel_name);
  }
  return KernelHandle();
}

std::vector<KernelHandle> KernelLoader::load_kernels(
    const std::vector<std::pair<std::string, std::string>>& kernel_configs) {
  auto& registry = KernelRegistry::get_instance();
  std::vector<KernelHandle> handles;

  if (registry.register_kernels(kernel_configs)) {
    for (const auto& config : kernel_configs) {
      handles.push_back(registry.get_kernel_handle(config.first));
    }
  }

  return handles;
}

KernelHandle KernelLoader::get_kernel(const std::string& kernel_name) {
  auto& registry = KernelRegistry::get_instance();
  return registry.get_kernel_handle(kernel_name);
}

bool KernelLoader::get_kernel_workspace_config(const std::string& kernel_name,
                                               int64_t& workspace_size,
                                               int64_t& lock_init_value,
                                               int64_t& lock_num) {
  auto& registry = KernelRegistry::get_instance();
  return registry.get_kernel_workspace_config(
      kernel_name, workspace_size, lock_init_value, lock_num);
}

int KernelLoader::load_kernels_from_directory(
    const std::string& directory_path) {
  if (!std::filesystem::exists(directory_path)) {
    LOG(ERROR) << "Directory does not exist: " << directory_path;
    return -1;
  }

  if (!std::filesystem::is_directory(directory_path)) {
    LOG(ERROR) << "Path is not a directory: " << directory_path;
    return -1;
  }

  LOG(INFO) << "Scanning directory for kernel binaries: " << directory_path;

  std::vector<std::pair<std::string, std::string>> kernel_configs;

  for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    std::string file_path = entry.path().string();
    std::string file_name = entry.path().filename().string();
    std::string extension = entry.path().extension().string();

    if (extension != ".npubin") {
      continue;
    }

    std::filesystem::path json_path_obj = entry.path();
    json_path_obj.replace_extension(".json");
    std::string json_path = json_path_obj.string();

    size_t last_dot = file_name.find_last_of('.');
    std::string kernel_name =
        (last_dot != std::string::npos) ? file_name.substr(0, last_dot)
                                        : file_name;

    if (std::filesystem::exists(json_path)) {
      std::ifstream json_file(json_path);
      if (json_file.is_open()) {
        nlohmann::json j =
            nlohmann::json::parse(json_file, nullptr, false, true);
        json_file.close();

        if (!j.is_discarded() && j.contains("kernel_name") &&
            j["kernel_name"].is_string()) {
          kernel_name = j["kernel_name"].get<std::string>();
          LOG(INFO) << "Found kernel '" << kernel_name
                    << "' from JSON: " << json_path;
        } else {
          LOG(INFO)
              << "JSON file exists but no valid kernel_name, using filename: "
              << kernel_name;
        }
      } else {
        LOG(WARNING) << "Cannot open JSON file: " << json_path
                     << ", using filename as kernel name";
      }
    } else {
      LOG(INFO) << "No JSON file found for " << file_name
                << ", using filename as kernel name: " << kernel_name;
    }

    kernel_configs.emplace_back(kernel_name, file_path);
  }

  if (kernel_configs.empty()) {
    LOG(INFO) << "No kernel binary files found in directory: " << directory_path;
    return 0;
  }

  LOG(INFO) << "Found " << kernel_configs.size() << " kernel binary file(s)";

  auto& registry = KernelRegistry::get_instance();
  if (registry.register_kernels(kernel_configs)) {
    LOG(INFO) << "Successfully registered " << kernel_configs.size()
              << " kernel(s)";
    return static_cast<int>(kernel_configs.size());
  }

  LOG(ERROR) << "Failed to register some kernels";
  return -1;
}

void KernelLoader::cleanup() {
  auto& registry = KernelRegistry::get_instance();
  registry.cleanup();
}

}  // namespace xllm::kernel::npu


