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

#include "qwen3_next_gated_delta_net.h"

#include <glog/logging.h>

#include <tuple>

namespace xllm {
namespace layer {

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                                       const QuantArgs& quant_args,
                                       const ParallelArgs& parallel_args,
                                       const torch::TensorOptions& options) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim(); 
  k_size_ = num_k_heads_ * head_k_dim_; 
  v_size_ = num_v_heads_ * head_v_dim_;

  bool has_bias = args.attention_bias();

  // 0. QKVZ parallel linear
  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                k_size_ * 2 + v_size_,
                                                /*bias=*/has_bias,
                                                /*gather_output=*/false,
                                                quant_args,
                                                parallel_args,
                                                options));


  // 1. QKVZ parallel linear
  qkvz_proj_ = register_module("in_proj_qkvz",
                                ColumnParallelLinear(args.hidden_size(),
                                                    k_size_ * 2 + v_size_ * 2,
                                                    /*bias=*/has_bias,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));
  // 2. Output projection
  ba_proj_ = register_module("in_proj_ba",
                              ColumnParallelLinear(args.hidden_size(),
                                                  num_k_heads_ * 2,
                                                  /*bias=*/has_bias,
                                                  /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));

  // 3. Output projection
  o_proj_ = register_module("out_proj",
                            RowParallelLinear(v_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args,
                                              options));

  // 4. RMSNorm
  norm_ = register_module("norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));

}

torch::Tensor Qwen3NextGatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // Implementation needed
  return torch::Tensor();
}

void Qwen3NextGatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  qkvz_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_qkvz."));
  ba_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_ba."));
  conv1d_->load_state_dict(state_dict.get_dict_with_prefix("conv1d."));
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
