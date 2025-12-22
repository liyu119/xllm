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
    torch::Tensor& x,
    torch::Tensor& conv_state,
    torch::Tensor& weight,
    torch::Tensor& bias,
    bool activation,
    const std::optional<torch::Tensor>& cache_seqlens,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc,
    int32_t max_query_len,
    const std::optional<torch::Tensor>& intermediate_conv_window,
    int32_t pad_slot_id,
    bool validate_data) {
    if (query_start_loc.has_value()) {
        LOG(ERROR) << "Current op does not support non-empty values for query_start_loc.";
        return torch::zeros(1);
    }
    bool unsqueeze = x.dim() == 2;
    if (unsqueeze) {
        x = x.unsqueeze(-1);
    }
    torch::Tensor out = torch::empty(
        x.sizes(), torch::TensorOptions().dtype(x.dtype()).device(x.device()));
    int32_t batch, dim, seqlen, width, num_cache_lines, state_len = 0;
    batch = x.size(0);
    dim = x.size(1);
    seqlen = x.size(2);
    width = weight.size(1);
    num_cache_lines = conv_state.size(0);
    state_len = conv_state.size(1);

    auto npu_stream = c10_npu::getCurrentNPUStream();
    rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

    void* x_ptr = x.data_ptr();
    void* conv_state_ptr = conv_state.data_ptr();
    void* weight_ptr = weight.data_ptr();
    void* conv_state_indices_ptr = conv_state_indices.has_value() ? conv_state_indices.value().data_ptr() : nullptr;
    void* out_ptr = out.data_ptr();

    int32_t gridX, gridY, gridZ = 1;
    void* workspace_addr = nullptr;
    void* sync_block_lock = nullptr;
    rtError_t ret;
    if (!cache_seqlens.has_value() && !num_accepted_tokens.has_value()) {
        gridX = batch;
        ret = launchers::_causal_conv1d_update_kernel_no_cache_len_no_mtp(
            stream,
            gridX,
            gridY,
            gridZ,
            workspace_addr,
            sync_block_lock,
            x_ptr,
            conv_state_ptr,
            weight_ptr,
            conv_state_indices_ptr,
            out_ptr,
            pad_slot_id);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to setup workspace and sync block lock for kernel "
            << "_causal_conv1d_update_kernel_no_cache_len_no_mtp" << " : error=" << ret;
            return out;
        }
    }
    return out;
}

}  // namespace xllm::kernel::npu