/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <torch/torch.h>

#include <optional>

#include "./custom_functions_npu/AtbCommon.h"
#include "./torch_api/triton_ops_api.h"

namespace xllm::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         torch::Tensor& value,
                         torch::Tensor& k_cache,
                         torch::Tensor& v_cache,
                         const torch::Tensor& slot_mapping);

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& mask,
                   const torch::Tensor& seq_len,
                   float scale,
                   torch::Tensor& output);

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  float scale,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  torch::Tensor& output);

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias);

torch::Tensor active(const torch::Tensor& input, const std::string& act_mode);

torch::Tensor fused_layernorm(const torch::Tensor& input,
                              const torch::Tensor& weight,
                              double eps,
                              const std::string& mode);

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& cos_sin_cache,
                  const torch::Tensor& positions);

torch::Tensor fused_moe(const torch::Tensor& hidden_states,
                        const torch::Tensor& gating_output,
                        const torch::Tensor& w1,
                        const torch::Tensor& w2,
                        const std::optional<torch::Tensor>& bias1,
                        const std::optional<torch::Tensor>& bias2,
                        const std::optional<torch::Tensor>& residual,
                        const std::optional<torch::Tensor>& input_smooth,
                        const std::optional<torch::Tensor>& act_smooth,
                        const std::optional<torch::Tensor>& w1_scale,
                        const std::optional<torch::Tensor>& w2_scale,
                        const std::optional<torch::Tensor>& e_score_correction_bias,
                        int topk,
                        bool renormalize,
                        bool gated,
                        const std::string& act_mode,
                        const std::string& scoring_func,
                        int num_expert_group,
                        int topk_group,
                        double route_scale,
                        int start_expert_id,
                        int block_n,
                        bool avg_moe,
                        const std::optional<torch::Tensor>& class_reduce_weight,
                        const std::optional<torch::Tensor>& class_expert_id,
                        const std::optional<torch::List<int64_t>>& w1_quant_flag,
                        const std::optional<torch::List<int64_t>>& w2_quant_flag,
                        int world_size,
                        int shared_expert_num,
                        const std::string& parallel_mode);
}  // namespace xllm::kernel::npu
