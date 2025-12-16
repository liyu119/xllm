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

namespace xllm::kernel::npu {

class KernelHandle {
 public:
  KernelHandle() = default;
  KernelHandle(const std::string& kernel_name, const char* handle);

  const char* get() const { return handle_; }
  operator const char*() const { return handle_; }

  bool is_valid() const { return handle_ != nullptr; }
  const std::string& name() const { return kernel_name_; }

 private:
  std::string kernel_name_;
  const char* handle_{nullptr};
};

}  // namespace xllm::kernel::npu


