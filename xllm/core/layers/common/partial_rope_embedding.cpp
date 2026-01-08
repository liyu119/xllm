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

#include "partial_rotary_embedding.h"

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

PartialRotaryEmbeddingImpl::PartialRotaryEmbeddingImpl(int64_t rotary_dim,
                                                       int64_t max_position_embeddings,
                                                       int64_t rope_theta,
                                                       int64_t head_size,
                                                       bool is_neox_style,
                                                       bool interleaved,
                                                       const torch::TensorOptions& options)
    : head_size_(head_size),
      rotary_dim_(rotary_dim),
      is_neox_style_(is_neox_style),
      interleaved_(interleaved) { 
  auto inv_freq = rotary::compute_inv_freq(rotary_dim, rope_theta, options);
  const auto cos_sin = rotary::compute_cos_sin_cache(
      rotary_dim, max_position_embeddings, interleaved, inv_freq, options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin);
}

void PartialRotaryEmbeddingImpl::forward(const torch::Tensor& positions,
                                  torch::Tensor& q,
                                  torch::Tensor& k) {
  xllm::kernel::PartialRotaryEmbedding partial_rotary_params;
  partial_rotary_params.positions = positions;
  partial_rotary_params.q = q;
  partial_rotary_params.k = k;
  partial_rotary_params.head_size = head_size_;
  partial_rotary_params.rotary_dim = rotary_dim_;
  partial_rotary_params.cos_sin = cos_sin_cache_;
  partial_rotary_params.is_neox_style = is_neox_style_;
  auto [q_rot, k_rot] = partial_rotary_embedding(params);

  q = q_rot;
  k = k_rot;
}

}  // namespace layer
}  // namespace xllm

