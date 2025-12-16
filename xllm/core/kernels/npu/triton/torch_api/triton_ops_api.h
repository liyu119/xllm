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

#include <optional>
#include <cmath>
#include <limits>
#include <algorithm>

#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>
 
#include "experiment/runtime/runtime/rt.h"
#include "kernel_launchers.h"
#include "utils.h"

namespace xllm::kernel::npu {
 

std::pair<torch::Tensor, torch::Tensor> npu_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta = 1.0f,
    float threshold = 20.0f);

std::pair<torch::Tensor, torch::Tensor> npu_fused_recurrent_gated_delta_rule(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const std::optional<torch::Tensor>& beta = std::nullopt,
    const std::optional<float>& scale = std::nullopt,
    const std::optional<torch::Tensor>& initial_state = std::nullopt,
    bool inplace_final_state = true,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& ssm_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt,
    bool use_qk_l2norm_in_kernel = false);
 
torch::Tensor npu_causal_conv1d_update(
    const torch::Tensor& x,
    const torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool activation = true, 
    const std::optional<torch::Tensor>& conv_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& num_accepted_tokens = std::nullopt,
    const std::optional<torch::Tensor>& query_start_loc = std::nullopt,
    int32_t max_query_len = -1,
    int32_t pad_slot_id = -1,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token = std::nullopt,
    const std::optional<torch::Tensor>& initial_state_idx = std::nullopt,
    bool validate_data = false);

 }  // namespace xllm::kernel::npu
 