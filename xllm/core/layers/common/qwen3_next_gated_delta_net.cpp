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
#include "xllm/core/kernels/ops_api.h"
#include <tuple>
#include <unordered_map>

namespace xllm {
namespace layer {

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                                       const QuantArgs& quant_args,
                                       const ParallelArgs& parallel_args,
                                       const torch::TensorOptions& options) {
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  tp_size_ = parallel_args.tp_group_->world_size();  
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
                                                  num_v_heads_ * 2,
                                                  /*bias=*/has_bias,
                                                  /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));

  dt_bias_ = register_parameter("dt_bias", torch::ones({num_v_heads_ / tp_size_}, dtype));                                                  
  A_log_ = register_parameter("A_log", torch::empty({num_v_heads_ / tp_size_}, torch::kFloat32));
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
  // 1. qkvz projection
  auto qkvz = qkvz_proj_->forward(hidden_states);
  auto [q, k, v, z] = process_qkvz_tensor(qkvz);

  // 2. ba projection
  auto ba = ba_proj_->forward(hidden_states);
  auto [b, a] = process_ba_tensor(ba);

  auto rearrange_merge = [](const torch::Tensor& t) {
    TORCH_CHECK(t.dim() >= 2, "Tensor must have at least 2 dims! but got ", t.dim());
    
    std::vector<int64_t> new_shape;
    int64_t tensor_dim = t.dim();

    // 2. 安全切片：取除最后 2 维外的所有维度（用正数索引替代 -2）
    int64_t slice_end = tensor_dim - 2; // 正数索引，避免无符号数解析坑
    if (slice_end > 0) { // 仅当有前置维度时才切片（≥3 维）
        auto valid_slice = t.sizes().slice(0, slice_end);
        new_shape = std::vector<int64_t>(valid_slice.begin(), valid_slice.end());
    }

    int64_t last_two_dim = t.size(tensor_dim - 2) * t.size(tensor_dim - 1);
    new_shape.push_back(last_two_dim);

    int64_t new_total_elems = 1;
    for (auto d : new_shape) {
        TORCH_CHECK(d > 0, "Invalid dimension 0 in new shape: ", new_shape);
        new_total_elems *= d;
    }

    return t.reshape(new_shape);
  };

  q = rearrange_merge(q);
  k = rearrange_merge(k);
  v = rearrange_merge(v);

  int64_t concat_dim = q.dim() - 1; 
  torch::Tensor mixed_qkv = torch::cat({q, k, v}, concat_dim);
  
  // 3. core attention
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  if (attn_metadata.is_prefill) {
    // Implement causal_conv1d_fn for prefill stage using PyTorch native operations
    // This is equivalent to: mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    auto conv_weight = conv1d_.weight.squeeze(1); // Remove the singleton dimension
    auto conv_bias = conv1d_.bias;
    
    // Apply 1D convolution
    // PyTorch conv1d input format: (batch, channels, sequence_length)
    // Weight format: (out_channels, in_channels/groups, kernel_size)
    auto conv_output = torch::conv1d(mixed_qkv, conv_weight, conv_bias, 
                                    /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/mixed_qkv.size(1));
    
    // Apply SiLU activation
    mixed_qkv = torch::silu(conv_output);

    //g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta = torch::sigmoid(b);
    torch::Tensor A_log_exp = A_log_float.exp();
    torch::Tensor a_float = a.to(torch::kFloat32);
    torch::Tensor a_plus_dt = a_float + dt_bias;
    torch::Tensor softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0) 
    );
    torch::Tensor g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();
  } else {
    xllm::kernel::CausalConv1dUpdateParams params;
    params.x = mixed_qkv;
    params.conv_state = conv_cache;
    params.weight = conv1d_.weight;
    params.bias = conv1d_.bias;
    params.activation = true;
    mixed_qkv = xllm::kernel::causal_conv1d_update(params);
    
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_;
    gdn_params.a = a;
    gdn_params.b = b;
    gdn_params.dt_bias = dt_bias_;
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    auto [g, beta] = xllm::kernel::fused_gdn_gating(gdn_params);
  }
  
  // Get dimensions
  int64_t batch_size = mixed_qkv.size(0);
  int64_t sequence_length = mixed_qkv.size(2);
  
  // Split sizes for q, k, v
  std::vector<int64_t> split_sizes = {k_size_, k_size_, v_size_};
  auto qkv_split = torch::split(mixed_qkv, split_sizes, 1);
  
  // Extract q, k, v from the processed mixed_qkv tensor
  auto processed_q = qkv_split[0];  // Shape: [batch_size, k_size_, sequence_length]
  auto processed_k = qkv_split[1];  // Shape: [batch_size, k_size_, sequence_length]
  auto processed_v = qkv_split[2];  // Shape: [batch_size, v_size_, sequence_length]
  
  // Reshape q, k to [1, batch_size, num_k_heads_/tp_size_, head_k_dim_]
  // Reshape v to [1, batch_size, num_v_heads_/tp_size_, head_v_dim_]
  processed_q = processed_q.view({1, batch_size, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view({1, batch_size, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view({1, batch_size, num_v_heads_ / tp_size_, head_v_dim_});
  
  // Assign the processed tensors back to q, k, v for downstream use
  q = processed_q;
  k = processed_k;
  v = processed_v;

  // 4. output projection
  auto attn_output = o_proj_->forward(rearrange_merge(z));
  return attn_output; 
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_qkvz_tensor(const torch::Tensor& qkvz) {
  

  //std::vector<int64_t> new_tensor_shape_qkvz = [&]() {
  //  std::vector<int64_t> dims(qkvz.size(0));
  //  dims.push_back(num_k_heads_ / tp_size_);
  // dims.push_back(head_k_dim_ + head_k_dim_ + (head_v_dim_ + head_v_dim_) * num_v_heads_ / num_k_heads_);
  //  return dims;
  //}();

  // 完整代码片段
  std::vector<int64_t> new_tensor_shape_qkvz = [&]() {
    std::vector<int64_t> dims;
    int64_t total_elems = qkvz.numel();

    if (qkvz.dim() >= 3) {
        dims = std::vector<int64_t>(
            qkvz.sizes().slice(0, -1).begin(),
            qkvz.sizes().slice(0, -1).end()
        );
    } else if (qkvz.dim() == 2) {
        dims.push_back(qkvz.size(0));
    } else {
        dims.push_back(qkvz.size(0));
    }

    int64_t dim1 = num_k_heads_ / tp_size_;
    int64_t dim2 = head_k_dim_ + head_k_dim_ + (head_v_dim_ + head_v_dim_) * num_v_heads_ / num_k_heads_;
    dims.push_back(dim1);
    dims.push_back(dim2);

    return dims;
  }();

  int64_t new_total = 1;
  for (auto d : new_tensor_shape_qkvz) new_total *= d;

  torch::Tensor qkvz_reshaped = qkvz.reshape(new_tensor_shape_qkvz);
  
  auto reshaped_qkvz = qkvz.view(new_tensor_shape_qkvz);
  
  auto qkvz_split = torch::split(reshaped_qkvz, 
    {head_k_dim_, head_k_dim_, 
     num_v_heads_ * head_v_dim_ / num_k_heads_, 
     head_v_dim_ * num_v_heads_ / num_k_heads_}, 2);
     
  auto q = qkvz_split[0].contiguous();
  auto k = qkvz_split[1].contiguous();
  auto v = qkvz_split[2].contiguous();
  auto z = qkvz_split[3].contiguous();

  v = v.view({z.size(0), -1, head_v_dim_});
  z = z.view({z.size(0), -1, head_v_dim_});

  return std::make_tuple(q, k, v, z);
}


std::tuple<torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_ba_tensor(const torch::Tensor& ba) {
  

  std::vector<int64_t> new_tensor_shape_ba = [&]() {
    std::vector<int64_t> dims;
    dims.push_back(ba.size(0));
    int64_t dim1 = num_k_heads_ / tp_size_;
    int64_t dim2 = 2 * num_v_heads_ / num_k_heads_;
    dims.push_back(dim1);
    dims.push_back(dim2);
    return dims;
  }();

  auto reshaped_ba = ba.view(new_tensor_shape_ba);
  auto ba_split = torch::split(reshaped_ba, 
    {num_v_heads_ / num_k_heads_, num_v_heads_ / num_k_heads_}, 2);
     
  auto b = ba_split[0].contiguous();
  auto a = ba_split[1].contiguous();

  b = b.reshape({b.size(0), num_v_heads_ / tp_size_});
  a = a.reshape({a.size(0), num_v_heads_ / tp_size_});
  
  return std::make_tuple(b, a);
}

void Qwen3NextGatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  qkvz_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_qkvz."));
  ba_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_ba."));
  
  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(StateDict({{"weight", w.squeeze(1)}}));
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
