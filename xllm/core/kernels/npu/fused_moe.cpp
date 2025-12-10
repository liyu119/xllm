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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"

namespace xllm::kernel::npu {

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
                        const std::string& parallel_mode) {
  // NPU FusedMoE implementation using torch_npu composition
  // Implements MoE routing and expert computation via basic torch operations

  auto ori_input_shape = hidden_states.sizes();
  auto hidden_states_2d = hidden_states.reshape({-1, hidden_states.size(-1)});
  auto gating_output_2d = gating_output.reshape({-1, gating_output.size(-1)});
  int64_t num_tokens = hidden_states_2d.size(0);
  int64_t hidden_size = hidden_states_2d.size(1);
  int64_t num_experts = gating_output_2d.size(1);

  // Step 1: Routing - Apply softmax/sigmoid based on scoring_func
  torch::Tensor routing_weights;
  if (scoring_func == "softmax") {
    routing_weights = torch::softmax(gating_output_2d * route_scale, -1);
  } else if (scoring_func == "sigmoid") {
    routing_weights = torch::sigmoid(gating_output_2d * route_scale);
  } else {
    throw std::runtime_error(
        "NPU fused_moe: unsupported scoring_func: " + scoring_func);
  }

  // Step 2: Select top-k experts per token
  auto [selected_weights, expert_indices] =
      torch::topk(routing_weights, topk, -1, true);

  if (renormalize) {
    selected_weights = selected_weights /
                       (selected_weights.sum(dim=-1, true) + 1e-6);
  }

  // Step 3: Dispatch tokens to experts
  // Create expert gate matrix: [num_tokens, topk]
  torch::Tensor gate_output = selected_weights;

  // Step 4: Compute MLP output for each expert
  // w1: [num_experts, hidden_size, intermediate_size]
  // Reshape for batch GEMM
  auto tokens_per_expert = num_tokens * topk;
  auto intermediate_size = w1.size(2);

  // Expand hidden_states based on expert assignment
  auto hidden_expanded = hidden_states_2d.unsqueeze(1).expand(
      {num_tokens, topk, hidden_size});
  hidden_expanded = hidden_expanded.reshape({tokens_per_expert, hidden_size});

  // Expand expert weights
  auto w1_expanded = w1.index_select(0, expert_indices.reshape({-1}));
  auto w2_expanded = w2.index_select(0, expert_indices.reshape({-1}));

  // First GEMM: hidden @ w1 -> intermediate
  auto intermediate = torch::matmul(hidden_expanded, w1_expanded.transpose(-2, -1));

  // Apply activation
  if (act_mode == "silu") {
    intermediate = torch::silu(intermediate);
  } else if (act_mode == "gelu") {
    intermediate = torch::nn::functional::gelu(intermediate);
  } else if (act_mode == "relu") {
    intermediate = torch::relu(intermediate);
  } else {
    throw std::runtime_error(
        "NPU fused_moe: unsupported activation: " + act_mode);
  }

  // Second GEMM: intermediate @ w2 -> output
  auto moe_output = torch::matmul(intermediate, w2_expanded.transpose(-2, -1));
  moe_output = moe_output.reshape({num_tokens, topk, hidden_size});

  // Step 5: Combine expert outputs with routing weights
  auto gate_reshaped = gate_output.unsqueeze(-1);  // [num_tokens, topk, 1]
  auto weighted_output = moe_output * gate_reshaped;
  auto final_output = weighted_output.sum(dim=1);  // [num_tokens, hidden_size]

  // Step 6: Add residual connection if provided
  if (residual.has_value()) {
    auto residual_2d = residual.value().reshape({-1, hidden_size});
    final_output = final_output + residual_2d;
  }

  // Reshape back to original shape
  return final_output.reshape(ori_input_shape);
}

}  // namespace xllm::kernel::npu
