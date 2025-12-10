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

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class RmsNormGatedImpl : public torch::nn::Module {
 public:
  RmsNormGatedImpl(int64_t dim,
                   double eps,
                   const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor& input);

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t norm_dim_;
  double eps_;
};

class RmsNormGated : public torch::nn::ModuleHolder<RmsNormGatedImpl> {
 public:
  using torch::nn::ModuleHolder<RmsNormGatedImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RmsNormGatedImpl;

  RmsNormGated(int64_t dim, double eps, const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<RmsNormGatedImpl>(dim, eps, options)) {}
};

}  // namespace layer
}  // namespace xllm