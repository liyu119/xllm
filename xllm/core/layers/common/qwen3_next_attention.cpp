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

#include "qwen3_next_attention.h"

#include <glog/logging.h>

#include <tuple>

namespace xllm {
namespace layer {

Qwen3NextAttentionImpl::Qwen3NextAttentionImpl(const ModelArgs& args,
                                               const QuantArgs& quant_args,
                                               const ParallelArgs& parallel_args,
                                               const torch::TensorOptions& options) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());

  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  CHECK(total_num_kv_heads % tp_size == 0);
  num_kv_heads_ = total_num_kv_heads / tp_size;
  num_kv_head_replicas_ = num_heads_ / num_kv_heads_;

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  // 1. QKV linear
  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                args.attn_output_gate ? num_heads_ * 2 : num_heads_,  
                                                num_kv_heads_,
                                                args.head_dim(),
                                                num_kv_head_replicas_,
                                                /*bias=*/args.attention_bias(),
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options));

  // 2. O proj
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(q_size_,
                                              args.hidden_size(),
                                              /*bias=*/args.attention_bias(),
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args,
                                              options));

  // 3. Q norm
  q_norm_ = register_module("q_norm",
                            RmsNorm(head_dim_, args.rms_norm_eps(), options));

  // 4. K norm
  k_norm_ = register_module("k_norm",
                            RmsNorm(head_dim_, args.rms_norm_eps(), options));
  
  // 5. Rotary embedding
  rotary_emb_ = register_module(
      "rotary_emb",
      RotaryEmbedding(head_dim_,
                      args.max_position_embeddings(),
                      args.rope_theta(),
                      /*interleaved=*/false,
                      options));

  // 6. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor Qwen3NextAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);

  // Get model args to check attn_output_gate
  // Since we don't have direct access to model args here, we'll infer from tensor shapes
  bool attn_output_gate = args.attn_output_gate;
  
  torch::Tensor q, k, v;
  torch::Tensor gate_part; // Declare gate_part outside the conditional block
  
  if (attn_output_gate) {
    // Split qkv for attn_output_gate case: [q_size*2, kv_size, kv_size]
    auto q_gate = qkv.slice(/*dim=*/-1, 0, q_size_ * 2);
    k = qkv.slice(/*dim=*/-1, q_size_ * 2, q_size_ * 2 + kv_size_);
    v = qkv.slice(/*dim=*/-1, q_size_ * 2 + kv_size_, q_size_ * 2 + kv_size_ * 2);
    
    // Split q_gate into q and gate
    q = q_gate.slice(/*dim=*/-1, 0, q_size_);
    gate_part = q_gate.slice(/*dim=*/-1, q_size_, q_size_ * 2);
  } else {
    // Normal case: [q_size, kv_size, kv_size]
    q = qkv.slice(/*dim=*/-1, 0, q_size_);
    k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
    v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
  }

  const int64_t T = q.size(0);

  // 2. q-norm
  q = q_norm_->forward(q);

  // 3. k-norm
  k = k_norm_->forward(k);

  // 4. rope
  rotary_emb_->forward(q,
                       k,
                       positions,
                       attn_metadata.query_start_loc,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);
  q = q.view({T, q_size_});
  k = k.view({T, kv_size_});

  // 5. store k/v cache and do attention
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  
  // 6. Apply attn_output_gate if enabled
  if (attn_output_gate) {
    // Reshape gate to match out dimensions
    auto gate_shape = gate_part.sizes().vec();
    gate_shape.pop_back();
    gate_shape.pop_back();
    gate_shape.push_back(-1);
    
    auto gate = gate_part.view(gate_shape);
    // Apply sigmoid activation to gate
    gate = torch::sigmoid(gate);
    // Apply gating to attention output
    out = out * gate;
  }

  // 7. output projection
  return o_proj_->forward(out);
}

void Qwen3NextAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict);
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
