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

#include "triton_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor npu_causal_conv1d_update(
    const torch::Tensor& x,
    const torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool activation, 
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc,
    int32_t max_query_len,
    int32_t pad_slot_id,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx,
    bool validate_data) {
    
    torch::Tensor out = torch::empty(
        x.sizes(), torch::TensorOptions().dtype(x.dtype()).device(x.device()));
    void* workspace_addr = nullptr;
    void* sync_block_lock = nullptr;

    return out;
}
    
}  // namespace xllm::kernel::npu
    