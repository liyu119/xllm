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

#include "acl/acl.h"

namespace xllm::kernel::npu {

inline aclError setup(const char* kernel_name,
                      void** workspace_addr,
                      void** sync_block_lock,
                      uint32_t block_num) {
    int64_t workspace_size = -1;
    int64_t lock_init_value = 0;
    int64_t lock_num = -1;
    KernelLoader::get_instance().get_kernel_workspace_config(kernel_name, 
        workspace_size, lock_init_value, lock_num);
    if (workspace_size > 0) {
        workspace_size *= block_num;
        auto ret = aclrtMalloc(workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to allocate workspace for kernel "
            << kernel_name << " : error=" << ret;
            return ret;
        }
    }
    if (lock_num > 0) {
        uint64_t sync_block_lock_size = lock_num * sizeof(int64_t);
        aclError ret = aclrtMalloc(sync_block_lock, sync_block_lock_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to allocate sync block lock for kernel "
            << kernel_name << " : error=" << ret;
            aclrtFree(*workspace_addr);
            *workspace_addr = nullptr;
            return ret;
        }
        std::vector<int64_t> lock_init_data(lock_num, lock_init_value);
        ret = aclrtMemcpy(*sync_block_lock, sync_block_lock_size, 
            reinterpret_cast<void*>(lock_init_data.data()), sync_block_lock_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to initialize sync block lock for kernel "
            << kernel_name << " : error=" << ret;
            aclrtFree(*workspace_addr);
            *workspace_addr = nullptr;
            aclrtFree(*sync_block_lock);
            *sync_block_lock = nullptr;
            return ret;
        }
    }
    return ACL_ERROR_NONE;
}

inline void cleanup(void* workspace_addr, void* sync_block_lock) {
    if (workspace_addr != nullptr) {
        aclrtFree(workspace_addr);
        workspace_addr = nullptr;
    }
    if (sync_block_lock != nullptr) {
        aclrtFree(sync_block_lock);
        sync_block_lock = nullptr;
    }
}

} // namespace xllm::kernel::npu