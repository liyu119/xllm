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

#include "kernel_registry.h"

#include <acl/acl.h>
#include <experiment/runtime/runtime/rt.h>

#include <filesystem>
#include <fstream>

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "kernel_handle.h"

namespace xllm::kernel::npu {

KernelRegistry& KernelRegistry::get_instance() {
  static KernelRegistry instance;
  return instance;
}

KernelRegistry::~KernelRegistry() { cleanup(); }

bool KernelRegistry::parse_json_config(const std::string& json_path,
                                       std::string& kernel_name,
                                       std::string& mix_mode,
                                       int64_t& workspace_size,
                                       int64_t& lock_init_value,
                                       int64_t& lock_num) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    LOG(WARNING) << "Cannot open JSON file: " << json_path;
    return false;
  }

  nlohmann::json j = nlohmann::json::parse(file, nullptr, false, true);
  file.close();

  if (j.is_discarded()) {
    LOG(ERROR) << "JSON parse error in " << json_path;
    return false;
  }

  if (j.contains("kernel_name") && j["kernel_name"].is_string()) {
    kernel_name = j["kernel_name"].get<std::string>();
  } else {
    LOG(WARNING) << "JSON file missing 'kernel_name' field: " << json_path;
    return false;
  }

  if (j.contains("mix_mode") && j["mix_mode"].is_string()) {
    mix_mode = j["mix_mode"].get<std::string>();
  } else {
    LOG(WARNING) << "JSON file missing 'mix_mode' field, using default 'aiv': "
                 << json_path;
    mix_mode = "aiv";
  }

  if (j.contains("workspace_size")) {
    if (j["workspace_size"].is_number_integer()) {
      workspace_size = j["workspace_size"].get<int64_t>();
    } else if (j["workspace_size"].is_null()) {
      workspace_size = -1;
    } else {
      LOG(WARNING) << "'workspace_size' is not an integer, using default -1";
      workspace_size = -1;
    }
  } else {
    workspace_size = -1;
  }

  if (j.contains("lock_init_value")) {
    if (j["lock_init_value"].is_number_integer()) {
      lock_init_value = j["lock_init_value"].get<int64_t>();
    } else if (j["lock_init_value"].is_null()) {
      lock_init_value = -1;
    } else {
      LOG(WARNING)
          << "'lock_init_value' is not an integer, using default -1";
      lock_init_value = -1;
    }
  } else {
    lock_init_value = -1;
  }

  if (j.contains("lock_num")) {
    if (j["lock_num"].is_number_integer()) {
      lock_num = j["lock_num"].get<int64_t>();
    } else if (j["lock_num"].is_null()) {
      lock_num = -1;
    } else {
      LOG(WARNING) << "'lock_num' is not an integer, using default -1";
      lock_num = -1;
    }
  } else {
    lock_num = -1;
  }

  return true;
}

bool KernelRegistry::register_kernel(const std::string& kernel_name,
                                     const std::string& binary_path) {
  if (kernel_infos_.find(kernel_name) != kernel_infos_.end()) {
    LOG(INFO) << "Kernel '" << kernel_name << "' is already registered";
    return true;
  }

  std::string json_path = binary_path;
  size_t last_dot = json_path.find_last_of('.');
  if (last_dot != std::string::npos) {
    json_path = json_path.substr(0, last_dot) + ".json";
  } else {
    json_path = binary_path + "/" + kernel_name + ".json";
  }

  std::string parsed_kernel_name = kernel_name;
  std::string mix_mode = "aiv";
  int64_t workspace_size = -1;
  int64_t lock_init_value = -1;
  int64_t lock_num = -1;

  if (std::filesystem::exists(json_path)) {
    if (parse_json_config(json_path,
                          parsed_kernel_name,
                          mix_mode,
                          workspace_size,
                          lock_init_value,
                          lock_num)) {
      LOG(INFO) << "Parsed JSON config: kernel_name=" << parsed_kernel_name
                << ", mix_mode=" << mix_mode
                << ", workspace_size=" << workspace_size
                << ", lock_init_value=" << lock_init_value
                << ", lock_num=" << lock_num;
    } else {
      LOG(WARNING)
          << "Failed to parse JSON config, using provided kernel name and "
             "default values";
    }
  } else {
    LOG(WARNING) << "JSON config file not found: " << json_path
                 << ", using provided kernel name and default values";
  }

  uint32_t file_size = 0;
  char* buffer = load_binary_file(binary_path + "/" + kernel_name + ".npubon", file_size);
  if (!buffer) {
    LOG(ERROR) << "Failed to load binary file for kernel '" << kernel_name
               << "'";
    return false;
  }

  KernelInfo info;
  info.name = kernel_name;
  info.buffer = buffer;
  info.persistent_func_name = parsed_kernel_name;
  info.mix_mode = mix_mode;
  info.workspace_size = workspace_size;
  info.lock_init_value = lock_init_value;
  info.lock_num = lock_num;

  void* stub_func = nullptr;
  if (!register_binary(info, file_size, &stub_func)) {
    LOG(ERROR) << "Failed to register binary for kernel '" << kernel_name
               << "'";
    delete[] buffer;
    return false;
  }

  info.stub_func = stub_func;
  kernel_infos_[kernel_name] = info;

  LOG(INFO) << "Successfully registered kernel '" << kernel_name
            << "' (function: " << parsed_kernel_name << ")";
  return true;
}

bool KernelRegistry::register_kernels(
    const std::vector<std::pair<std::string, std::string>>& kernel_configs) {
  bool all_success = true;
  size_t success_count = 0;
  for (const auto& config : kernel_configs) {
    if (register_kernel(config.first, config.second)) {
      success_count++;
    } else {
      all_success = false;
    }
  }
  LOG(INFO) << "Registered " << success_count << " out of "
            << kernel_configs.size() << " kernel(s).";
  return all_success;
}

KernelHandle KernelRegistry::get_kernel_handle(
    const std::string& kernel_name) const {
  auto it = kernel_infos_.find(kernel_name);
  if (it != kernel_infos_.end()) {
    return KernelHandle(kernel_name,
                        static_cast<const char*>(it->second.stub_func));
  }
  return KernelHandle();
}

bool KernelRegistry::is_kernel_registered(
    const std::string& kernel_name) const {
  return kernel_infos_.find(kernel_name) != kernel_infos_.end();
}

std::vector<std::string> KernelRegistry::get_registered_kernels() const {
  std::vector<std::string> kernels;
  kernels.reserve(kernel_infos_.size());
  for (const auto& pair : kernel_infos_) {
    kernels.push_back(pair.first);
  }
  return kernels;
}

bool KernelRegistry::get_kernel_workspace_config(
    const std::string& kernel_name,
    int64_t& workspace_size,
    int64_t& lock_init_value,
    int64_t& lock_num) const {
  auto it = kernel_infos_.find(kernel_name);
  if (it != kernel_infos_.end()) {
    workspace_size = it->second.workspace_size;
    lock_init_value = it->second.lock_init_value;
    lock_num = it->second.lock_num;
    return true;
  }
  return false;
}

char* KernelRegistry::load_binary_file(const std::string& file_path,
                                       uint32_t& file_size) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    LOG(ERROR) << "Cannot open binary file: " << file_path;
    return nullptr;
  }

  file_size = static_cast<uint32_t>(file.tellg());
  if (file_size == 0) {
    LOG(ERROR) << "Binary file is empty: " << file_path;
    file.close();
    return nullptr;
  }

  file.seekg(0, std::ios::beg);

  char* buffer = new (std::nothrow) char[file_size];
  if (!buffer) {
    LOG(ERROR) << "Failed to allocate memory for binary file: " << file_path
               << " (size: " << file_size << " bytes)";
    file.close();
    return nullptr;
  }

  file.read(buffer, file_size);
  if (file.gcount() != static_cast<std::streamsize>(file_size)) {
    LOG(ERROR) << "Failed to read complete binary file: " << file_path;
    delete[] buffer;
    file.close();
    return nullptr;
  }

  file.close();
  return buffer;
}

bool KernelRegistry::register_binary(KernelInfo& info,
                                     uint32_t binary_size,
                                     void** stub_func) {
  rtDevBinary_t binary;
  void* binHandle = nullptr;

  binary.data = info.buffer;
  binary.length = binary_size;
  if (info.mix_mode == "aiv") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else if (info.mix_mode == "aic") {
    // TODO when kernel type is aicube, magic value will be RT_DEV_BINARY_MAGIC_ELF_AICUBE
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  } else {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
  }
  binary.version = 0;

  rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
  if (rtRet != RT_ERROR_NONE) {
    LOG(ERROR) << "rtDevBinaryRegister failed for kernel '" << info.name
               << "': error=" << rtRet;
    return false;
  }

  rtRet = rtFunctionRegister(binHandle,
                             info.persistent_func_name.c_str(),
                             info.persistent_func_name.c_str(),
                             (void*)info.persistent_func_name.c_str(),
                             0);

  if (rtRet != RT_ERROR_NONE) {
    LOG(ERROR) << "rtFunctionRegister failed for kernel '" << info.name
               << "' with function name '" << info.persistent_func_name
               << "': error=" << rtRet;
    return false;
  }

  *stub_func = (void*)info.persistent_func_name.c_str();
  return true;
}

void KernelRegistry::cleanup() {
  size_t count = kernel_infos_.size();
  for (auto& [name, info] : kernel_infos_) {
    if (info.buffer) {
      delete[] info.buffer;
      info.buffer = nullptr;
    }
  }
  kernel_infos_.clear();
  LOG(INFO) << "KernelRegistry cleanup completed. Unregistered " << count
            << " kernel(s).";
}

}  // namespace xllm::kernel::npu


