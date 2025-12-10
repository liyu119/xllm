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

#include "rms_norm_gated.h"

#include <glog/logging.h>

#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

RmsNormGatedImpl::RmsNormGatedImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_buffer("weight", torch::ones({dim}, options));
}

torch::Tensor RmsNormGatedImpl::forward(torch::Tensor& input) {
  // For gated RMSNorm, we apply the standard RMSNorm formula
  // variance = mean(square(input))
  // normalized = input / sqrt(variance + eps)
  // output = weight * normalized
  
  auto input_dtype = input.dtype();
  input = input.to(torch::kFloat32);
  
  // Calculate RMS
  auto variance = torch::mean(torch::pow(input, 2), -1, true);
  auto normalized = input * torch::rsqrt(variance + eps_);
  
  // Apply weight and convert back to original dtype
  return (normalized * weight_).to(input_dtype);
}

void RmsNormGatedImpl::load_state_dict(const StateDict& state_dict) {
  load_tensor(state_dict, "weight", weight_);
}

}  // namespace layer
}  // namespace xllm