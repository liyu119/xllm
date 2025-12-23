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
// Include headers for kernel loading
#include "xllm/core/kernels/npu/triton/kernel_loader.h"
#include "xllm/core/kernels/npu/triton/test/test_utils.h"
#include <torch_npu/torch_npu.h>
#include <acl/acl.h>

#include <tuple>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace layer {


namespace {
torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
    return x / (x.norm(2, dim, true) + eps);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    c10::optional<torch::Tensor> initial_state = c10::nullopt,
    bool output_final_state = false,
    bool use_qk_l2norm_in_kernel = false) {
    
    auto initial_dtype = query.dtype();
    
    if (use_qk_l2norm_in_kernel) {
        query = l2norm(query, -1, 1e-6);
        key = l2norm(key, -1, 1e-6);
    }
    
    // 转换数据类型并调整维度 [B, S, H, D] -> [B, H, S, D]
    auto to_float32 = [](torch::Tensor x) {
        return x.transpose(1, 2).contiguous().to(torch::kFloat32);
    };
    
    query = to_float32(query);
    key = to_float32(key);
    value = to_float32(value);
    beta = to_float32(beta);
    g = to_float32(g);
    
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto sequence_length = query.size(2);
    auto k_head_dim = key.size(-1);
    auto v_head_dim = value.size(-1);
    
    // 计算需要 padding 的长度
    int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
    
    query = torch::nn::functional::pad(query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
    g = torch::nn::functional::pad(g, torch::nn::functional::PadFuncOptions({0, pad_size}));
    
    int64_t total_sequence_length = sequence_length + pad_size;
    float scale = 1.0 / std::sqrt(query.size(-1));
    query = query * scale;
    
    // 计算 v_beta 和 k_beta
    auto v_beta = value * beta.unsqueeze(-1);
    auto k_beta = key * beta.unsqueeze(-1);
    
    // 重塑为分块结构 [B, H, num_chunks, chunk_size, D]
    auto reshape_to_chunks = [chunk_size](torch::Tensor x) {
        auto shape = x.sizes();
        std::vector<int64_t> new_shape = {
            shape[0], shape[1], 
            shape[2] / chunk_size, chunk_size, 
            shape[3]
        };
        return x.reshape(new_shape);
    };
    
    query = reshape_to_chunks(query);
    key = reshape_to_chunks(key);
    value = reshape_to_chunks(value);
    k_beta = reshape_to_chunks(k_beta);
    v_beta = reshape_to_chunks(v_beta);
    
    // g 重塑 [B, H, num_chunks, chunk_size]
    auto g_shape = g.sizes();
    std::vector<int64_t> g_new_shape = {
        g_shape[0], g_shape[1], 
        g_shape[2] / chunk_size, chunk_size
    };
    g = g.reshape(g_new_shape);
    
    // 创建上三角掩码
    auto mask = torch::triu(
        torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())),
        0
    );
    
    // Chunk decay 计算
    g = g.cumsum(-1);
    auto decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().to(torch::kFloat32)).tril();
    
    // 注意力计算
    auto attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0.0);
    
    // 循环计算注意力矩阵
    for (int64_t i = 1; i < chunk_size; ++i) {
        auto row = attn.slice(-2, i, i+1).slice(-1, 0, i).clone();
        auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
        attn.slice(-2, i, i+1).slice(-1, 0, i) = row + (row.unsqueeze(-1) * sub).sum(-2);
    }
    
    attn = attn + torch::eye(chunk_size, torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
    value = attn @ v_beta;
    
    // 计算 k_cumdecay
    auto k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1));
    
    // 初始化状态
    torch::Tensor last_recurrent_state;
    if (!initial_state.has_value()) {
        last_recurrent_state = torch::zeros(
            {batch_size, num_heads, k_head_dim, v_head_dim},
            torch::TensorOptions().dtype(value.dtype()).device(value.device())
        );
    } else {
        last_recurrent_state = initial_state.value().to(value);
    }
    
    // 创建输出张量
    auto core_attn_out = torch::zeros_like(value);
    
    // 创建新的上三角掩码 (diagonal=1)
    mask = torch::triu(
        torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())),
        1
    );
    
    // 遍历每个 chunk
    int64_t num_chunks = total_sequence_length / chunk_size;
    for (int64_t i = 0; i < num_chunks; ++i) {
        auto q_i = query.select(2, i);
        auto k_i = key.select(2, i);
        auto v_i = value.select(2, i);
        
        // 计算注意力
        auto attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask.select(2, i)).masked_fill_(mask, 0.0);
        
        // 计算 v_prime
        auto v_prime = k_cumdecay.select(2, i) @ last_recurrent_state;
        
        // 计算 v_new
        auto v_new = v_i - v_prime;
        
        // 计算注意力中间结果
        auto attn_inter = (q_i * g.select(2, i).unsqueeze(-1).exp()) @ last_recurrent_state;
        
        // 计算核心注意力输出
        core_attn_out.select(2, i) = attn_inter + attn_i @ v_new;
        
        // 更新状态
        last_recurrent_state = 
            last_recurrent_state * g.select(2, i).select(-1, -1).unsqueeze(-1).unsqueeze(-1).exp() +
            (k_i * (g.select(2, i).select(-1, -1).unsqueeze(-1) - g.select(2, i)).exp().unsqueeze(-1))
                .transpose(-1, -2) @ v_new;
    }
    
    // 处理最终状态输出
    if (!output_final_state) {
        last_recurrent_state = torch::Tensor(); // 置为空张量
    }
    
    // 重塑输出并移除 padding
    auto core_attn_out_shape = core_attn_out.sizes();
    std::vector<int64_t> reshape_shape = {
        core_attn_out_shape[0], core_attn_out_shape[1],
        core_attn_out_shape[2] * core_attn_out_shape[3],
        core_attn_out_shape[4]
    };
    core_attn_out = core_attn_out.reshape(reshape_shape);
    
    // 裁剪掉 padding 部分
    core_attn_out = core_attn_out.slice(2, 0, sequence_length);
    
    // 恢复维度顺序和数据类型
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
    
    return std::make_tuple(core_attn_out, last_recurrent_state);

}
} // namespace

void Qwen3NextGatedDeltaNetImpl::load_triton_kernel(
    const std::string& kernel_name,
    const std::string& binary_filename) {
  
  try {
    // Initialize NPU if available
    torch_npu::init_npu("npu:0");
    
    // Get binary path
    std::string binary_path = GetKernelBinaryPath(binary_filename);
    
    // Load kernel using KernelLoader
    auto& loader = xllm::kernel::npu::KernelLoader::get_instance();
    auto handle = loader.get_kernel(kernel_name);
    if (!handle.is_valid()) {
      handle = loader.load_kernel(kernel_name, binary_path);
    }
    
    if (handle.is_valid()) {
      is_kernel_loaded_ = true;
      LOG(INFO) << "Successfully loaded triton kernel: " << kernel_name;
    } else {
      is_kernel_loaded_ = false;
      LOG(WARNING) << "Failed to load triton kernel: " << kernel_name
                   << " from " << binary_path;
    }
  } catch (const std::exception& e) {
    is_kernel_loaded_ = false;
    LOG(WARNING) << "Exception occurred while loading triton kernel: " << e.what();
  }
}

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
                                                /*bias=*/false,
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

  dt_bias_ = register_parameter("dt_bias", torch::ones({num_v_heads_ / tp_size_}, options));
  
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
  
  // Initialize kernel loading related members
  gdn_triton_kernel_name_ = "fused_gdn_gating_head8_kernel";
  gdn_triton_binary_filename_ = "fused_gdn_gating_head8_kernel.npubin";

  conv_triton_kernel_name_ = "_causal_conv1d_update_kernel_no_cache_len_no_mtp";
  conv_triton_binary_filename_ = "_causal_conv1d_update_kernel_no_cache_len_no_mtp.npubin";

  recurrent_triton_kernel_name_ = "fused_recurrent_gated_delta_rule_fwd_kernel";
  recurrent_triton_binary_filename_ = "fused_recurrent_gated_delta_rule_fwd_kernel.npubin";
  
  // Load the fused GDN gating kernel
  load_triton_kernel(gdn_triton_kernel_name_, gdn_triton_binary_filename_);
  load_triton_kernel(conv_triton_kernel_name_, conv_triton_binary_filename_);
  load_triton_kernel(recurrent_triton_kernel_name_, recurrent_triton_binary_filename_);
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

    int64_t slice_end = tensor_dim - 2; 
    if (slice_end > 0) { 
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
  mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1,2);
  int64_t seq_len = mixed_qkv.size(2); 
  // 3. core attention
  // torch::Tensor conv_cache = kv_cache.get_conv_cache();
  // torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  if (attn_metadata.is_prefill) {
    auto device = mixed_qkv.device();
    auto conv_weight = conv1d_->weight().unsqueeze(1).to(device); 
    //  auto conv_bias = None;
    std::cerr << "[PREFILL] mixed_qkv - Device: " << mixed_qkv.device() 
              << ", Dtype: " << mixed_qkv.dtype() 
              << ", Shape: " << mixed_qkv.sizes() << std::endl;
    std::cerr << "[PREFILL] conv_weight - Device: " << conv_weight.device() 
              << ", Dtype: " << conv_weight.dtype() 
              << ", Shape: " << conv_weight.sizes() << std::endl;

    auto conv_output = torch::conv1d(
        mixed_qkv,
        conv_weight,
        torch::Tensor(),  
        /*stride=*/std::vector<int64_t>{1},
        /*padding=*/std::vector<int64_t>{3},
        /*dilation=*/std::vector<int64_t>{1},
        /*groups=*/static_cast<int64_t>(mixed_qkv.size(1))
    );
    mixed_qkv = torch::silu(conv_output.slice(2,0,seq_len));

    std::cerr << "[PREFILL] after mixed_qkv - Device: " << mixed_qkv.device() 
              << ", Dtype: " << mixed_qkv.dtype() 
              << ", Shape: " << mixed_qkv.sizes() << std::endl;
    std::cerr << "[PREFILL] a_log - Device: " << A_log_.device()
              << ", Dtype: " << A_log_.dtype()
              << ", Shape: " << A_log_.sizes() << std::endl;
    std::cerr << "[PREFILL] dt_bias_ - Device: " << dt_bias_.device()
              << ", Dtype: " << dt_bias_.dtype()
              << ", Shape: " << dt_bias_.sizes() << std::endl;

    //g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    torch::Tensor beta = torch::sigmoid(b);
    torch::Tensor A_log_exp = A_log_.exp().to(device);
    torch::Tensor a_float = a.to(torch::kFloat32);
    torch::Tensor a_plus_dt = a_float + dt_bias_.to(device);
    torch::Tensor softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0) 
    );
    torch::Tensor g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();

    std::cerr << "[PREFILL] beta - Device: " << beta.device() 
              << ", Dtype: " << beta.dtype() 
              << ", Shape: " << beta.sizes() << std::endl;
    std::cerr << "[PREFILL] g - Device: " << g.device() 
              << ", Dtype: " << g.dtype() 
              << ", Shape: " << g.sizes() << std::endl;
    // Get dimensions and process mixed_qkv tensor
    auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
  
    auto [core_attn_out, last_recurrent_state] = torch_chunk_gated_delta_rule(processed_q, processed_k, processed_v, g, beta);
    // 4. output projection
    auto attn_output = o_proj_->forward(rearrange_merge(z));
    return attn_output; 

  } else {

    std::cerr << "decode " << std::endl;
    std::cerr << "mixed_qkv Shape: " << mixed_qkv.sizes() << std::endl;
    auto device = mixed_qkv.device();

    auto conv_weight_original = conv1d_->weight();
    std::cerr << "conv1d Weight Original Shape: " << conv_weight_original.sizes() << std::endl; // [2048,4]

    auto conv_weight = conv_weight_original.unsqueeze(1).contiguous().to(device, /*non_blocking=*/false);
    std::cerr << "conv1d Weight Expanded Shape: " << conv_weight.sizes() << std::endl; // [2048,1,4]

    xllm::kernel::CausalConv1dUpdateParams params;
    params.x = mixed_qkv;
    params.conv_state = torch::Tensor();
    params.weight = conv_weight;
    params.bias = torch::Tensor();
    params.activation = true;
    mixed_qkv = xllm::kernel::causal_conv1d_update(params);

    std::cerr << "[DECODE] after mixed_qkv - Device: " << mixed_qkv.device()
              << ", Dtype: " << mixed_qkv.dtype()
              << ", Shape: " << mixed_qkv.sizes() << std::endl;
    
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_.to(device);
    gdn_params.a = a;
    gdn_params.b = b;
    gdn_params.dt_bias = dt_bias_to(torch::kFloat32).to(device, /*non_blocking=*/false);
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    auto [g, beta] = xllm::kernel::fused_gdn_gating(gdn_params);
    std::cerr << "[DECODE] beta - Device: " << beta.device()
              << ", Dtype: " << beta.dtype()
              << ", Shape: " << beta.sizes() << std::endl;
    std::cerr << "[DECODE] g - Device: " << g.device()
              << ", Dtype: " << g.dtype()
              << ", Shape: " << g.sizes() << std::endl;

    // Get dimensions and process mixed_qkv tensor
    auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
    
    int64_t repeat_times = num_v_heads_ / num_k_heads_;
    if (repeat_times > 1) {
        processed_q = processed_q.repeat_interleave(repeat_times, 2);
        processed_k = processed_k.repeat_interleave(repeat_times, 2);
    }
    g = g.unsqueeze(0);
    beta = beta.unsqueeze(0);

    xllm::kernel::FusedRecurrentGatedDeltaRuleParams recurrent_gated_params;
    recurrent_gated_params.q = processed_q;
    recurrent_gated_params.k = processed_k;
    recurrent_gated_params.v = processed_v;
    recurrent_gated_params.g = g;
    recurrent_gated_params.beta = beta;
    recurrent_gated_params.scale = 1.0f;
    recurrent_gated_params.use_qk_l2norm_in_kernel = false;
    recurrent_gated_params.inplace_final_state = true;
    recurrent_gated_params.initial_state = torch::Tensor();
    recurrent_gated_params.cu_seqlens = torch::Tensor();
    recurrent_gated_params.num_accepted_tokens = torch::Tensor();
    auto [core_attn_out, last_recurrent_state] = xllm::kernel::fused_recurrent_gated_delta_rule(recurrent_gated_params);
  }
  


  // 4. output projection
  auto attn_output = o_proj_->forward(rearrange_merge(z));
  return attn_output; 
}

// Method to process the mixed_qkv tensor to extract and reshape q, k, v tensors
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) {
  // Get dimensions
  mixed_qkv = mixed_qkv.transpose(1,2);
  int64_t batch_size = mixed_qkv.size(0);
  int64_t sequence_length = mixed_qkv.size(1);
    
  // Split sizes for q, k, v
  std::vector<int64_t> split_sizes = {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto qkv_split = torch::split(mixed_qkv, split_sizes, 2);
  
  // Extract q, k, v from the processed mixed_qkv tensor
  auto processed_q = qkv_split[0];  // Shape: [batch_size, k_size_, sequence_length]
  auto processed_k = qkv_split[1];  // Shape: [batch_size, k_size_, sequence_length]
  auto processed_v = qkv_split[2];  // Shape: [batch_size, v_size_, sequence_length]
  
  // Reshape q, k to [1, batch_size, num_k_heads_/tp_size_, head_k_dim_]
  // Reshape v to [1, batch_size, num_v_heads_/tp_size_, head_v_dim_]
  processed_q = processed_q.view({batch_size, sequence_length, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view({batch_size, sequence_length, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view({batch_size, sequence_length, num_v_heads_ / tp_size_, head_v_dim_});
  
  return std::make_tuple(processed_q, processed_k, processed_v);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_qkvz_tensor(const torch::Tensor& qkvz) {
  
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
  if (!is_kernel_loaded_) {
    load_fused_gdn_gating_kernel();
  }
  
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
