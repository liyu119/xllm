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

#include <glog/logging.h>

#include <string>
#include <vector>

#include "experiment/runtime/runtime/rt.h"
#include "kernel_loader.h"
#include "torch_api/utils.h"

#define RT_ERROR_INVALID_VALUE -1
#define _DECLARE_STRUCT_FIELD(type, name) \
  type name __attribute__((aligned(sizeof(type))));
#define _DECLARE_PARAM(type, name) , type name
#define _DECLARE_ARG_VALUE(type, name) , name

#define REG_KERNEL_ARGS(kernel_name, ARG_LIST_MACRO)     \
    struct __attribute__((packed)) _##kernel_name##_Args { \
      void* ffts_addr __attribute__((aligned(8)));         \
      void* sync_block_lock __attribute__((aligned(8)));     \
      void* workspace_addr __attribute__((aligned(8)));    \
      ARG_LIST_MACRO(_DECLARE_STRUCT_FIELD)                \
      int32_t gridX __attribute__((aligned(4)));           \
      int32_t gridY __attribute__((aligned(4)));           \
      int32_t gridZ __attribute__((aligned(4)));           \
    };

#define REG_KERNEL_LAUNCHER(kernel_name, ARG_LIST_MACRO)                       \
static rtError_t kernel_name(                                                \
    rtStream_t stream,                                                       \
    int32_t gridX,                                                           \
    int32_t gridY,                                                           \
    int32_t gridZ,                                                           \
    void* workspace_addr,                                                    \
    void* sync_block_lock ARG_LIST_MACRO(_DECLARE_PARAM)) {                    \
    auto kernelHandle = xllm::kernel::npu::KernelLoader::get_instance().           \
                            get_kernel(#kernel_name);                          \
    if (!kernelHandle.is_valid()) {                                            \
      LOG(ERROR) << "Kernel '" << #kernel_name << "' is not registered";       \
      return RT_ERROR_INVALID_VALUE;                                           \
    }                                                                          \
                                                                              \
    uint32_t blockNum = gridX * gridY * gridZ;                                 \
                                                                              \
    void* ffts_addr = nullptr;                                                 \
    uint32_t ffts_len = 0;                                                     \
    rtError_t ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);        \
    if (ret != RT_ERROR_NONE) {                                                \
      LOG(ERROR) << "rtGetC2cCtrlAddr failed: " << ret;                        \
      return ret;                                                              \
    }                                                                          \
                                                                              \
    ret = xllm::kernel::npu::setup(#kernel_name, &workspace_addr,              \
      &sync_block_lock, blockNum);                                            \
    if (ret != ACL_ERROR_NONE) {                                               \
      LOG(ERROR) << "Failed to setup workspace and sync block lock for kernel " \
      << #kernel_name << " : error=" << ret;                                  \
      return ret;                                                              \
    }                                                                          \
    _##kernel_name##_Args args = {                                             \
        ffts_addr,                                                             \
        sync_block_lock,                                                         \
        workspace_addr ARG_LIST_MACRO(_DECLARE_ARG_VALUE),                    \
        gridX,                                                                 \
        gridY,                                                                 \
        gridZ};                                                                \
                                                                              \
    ret = rtKernelLaunch(kernelHandle.get(),                                   \
                        blockNum,                                             \
                        static_cast<void*>(&args),                            \
                        sizeof(_##kernel_name##_Args),                        \
                        nullptr,                                              \
                        stream);                                              \
                                                                              \
    if (ret != RT_ERROR_NONE) {                                                \
      LOG(ERROR) << "rtKernelLaunch failed for '" << #kernel_name              \
                << "': " << ret;                                              \
      xllm::kernel::npu::cleanup(workspace_addr, sync_block_lock);             \
      return ret;                                                              \
    }                                                                          \
    xllm::kernel::npu::cleanup(workspace_addr, sync_block_lock);               \
                                                                              \
    return RT_ERROR_NONE;                                                      \
}

 
 