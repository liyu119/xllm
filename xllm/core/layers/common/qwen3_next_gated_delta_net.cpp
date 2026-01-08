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
namespace xllm {
namespace layer {


namespace {
torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
    return x / (x.norm(2, dim, true) + eps);
}

std::tuple<torch::Tensor, torch::Tensor> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size = 64,
    std::optional<torch::Tensor> initial_state = std::nullopt,
    bool output_final_state = false,
    bool use_qk_l2norm_in_kernel = true) {
    
    auto initial_dtype = query.dtype();
    
    if (use_qk_l2norm_in_kernel) {
        query = l2norm(query, -1, 1e-6);
        key = l2norm(key, -1, 1e-6);
    }

    g = g.unsqueeze(0);
    beta = beta.unsqueeze(0);

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
    
    int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
    
    query = torch::nn::functional::pad(query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
    g = torch::nn::functional::pad(g, torch::nn::functional::PadFuncOptions({0, pad_size}));
    
    int64_t total_sequence_length = sequence_length + pad_size;
    float scale = 1.0 / std::sqrt(query.size(-1));
    query = query * scale;
    
    auto v_beta = value * beta.unsqueeze(-1);
    auto k_beta = key * beta.unsqueeze(-1);
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
    
    auto g_shape = g.sizes();
    std::vector<int64_t> g_new_shape = {
        g_shape[0], g_shape[1], 
        g_shape[2] / chunk_size, chunk_size
    };
    g = g.reshape(g_new_shape);
    auto mask = torch::triu(
        torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())),
        0
    );
    
    g = g.cumsum(-1);
    auto decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().to(torch::kFloat32)).tril();
    auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0.0);
    for (int64_t i = 1; i < chunk_size; ++i) {
        auto row = attn.slice(-2, i, i+1).slice(-1, 0, i).clone().contiguous();
        auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
        auto sub_unsq = sub.unsqueeze(-1).contiguous();
        auto row_unsq = row.unsqueeze(-1).contiguous();
        auto row_sub_mul = (row_unsq * sub_unsq).contiguous();
        auto row_sub_sum = row_sub_mul.sum(-2).contiguous();
        auto row_final = (row + row_sub_sum).contiguous();

        auto result_to_assign = row_final.slice(3, 0, 1);
        auto attn_new = attn.clone().contiguous();
        auto assign_target_new = attn_new.slice(-2, i, i+1).slice(-1, 0, i);

        assign_target_new.copy_(result_to_assign.contiguous());
        attn = attn_new;

    }
    
    attn = attn + torch::eye(chunk_size, torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
    value = torch::matmul(attn, v_beta);
    
    auto k_cumdecay = torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
    torch::Tensor last_recurrent_state;
    if (!initial_state.has_value()) {
        last_recurrent_state = torch::zeros(
            {batch_size, num_heads, k_head_dim, v_head_dim},
            torch::TensorOptions().dtype(value.dtype()).device(value.device())
        );
    } else {
        last_recurrent_state = initial_state.value().to(value);
    }
    auto core_attn_out = torch::zeros_like(value);
    
    mask = torch::triu(
        torch::ones({chunk_size, chunk_size}, torch::TensorOptions().dtype(torch::kBool).device(query.device())),
        1
    );
    
    int64_t num_chunks = total_sequence_length / chunk_size;
    for (int64_t i = 0; i < num_chunks; ++i) {
        auto q_i = query.select(2, i);
        auto k_i = key.select(2, i);
        auto v_i = value.select(2, i);
        
        auto attn_i = (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i)).masked_fill_(mask, 0.0);
        
        auto v_prime = torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
        
        auto v_new = v_i - v_prime;
        
        auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(), last_recurrent_state);
        
        core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
        
        last_recurrent_state = 
            last_recurrent_state * g.select(2, i).select(-1, -1).unsqueeze(-1).unsqueeze(-1).exp() +
            torch::matmul((k_i * (g.select(2, i).select(-1, -1).unsqueeze(-1) - g.select(2, i)).exp().unsqueeze(-1))
                .transpose(-1, -2), v_new);
    }
    
//    if (!output_final_state) {
//        last_recurrent_state = torch::Tensor();
//    }
    
    auto core_attn_out_shape = core_attn_out.sizes();
    std::vector<int64_t> reshape_shape = {
        core_attn_out_shape[0], core_attn_out_shape[1],
        core_attn_out_shape[2] * core_attn_out_shape[3],
        core_attn_out_shape[4]
    };
    core_attn_out = core_attn_out.reshape(reshape_shape);
    
    core_attn_out = core_attn_out.slice(2, 0, sequence_length);
    
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
    return std::make_tuple(core_attn_out, last_recurrent_state);

}
} // namespace

void Qwen3NextGatedDeltaNetImpl::load_triton_kernel(
    const std::string& kernel_name,
    const std::string& binary_filename) {
  // Set kernel name and binary filename

  try {
    // Initialize NPU if available
    torch_npu::init_npu("npu:0");

    // Get binary path
    std::string binary_path = xllm::kernel::npu::GetKernelBinaryPath(binary_filename);
    binary_path = "/export/home/weinan5/liyu/xllm_2/xllm/core/kernels/npu/triton/binary";
    // Load kernel using KernelLoader
    auto& loader = xllm::kernel::npu::KernelLoader::get_instance();
    auto handle = loader.get_kernel(kernel_name);
    if (!handle.is_valid()) {
      handle = loader.load_kernel(kernel_name, binary_path);
    }

    if (handle.is_valid()) {
      LOG(INFO) << "Successfully loaded fused GDN gating kernel: " << kernel_name;
    } else {
      LOG(WARNING) << "Failed to load fused GDN gating kernel: " << kernel_name
                   << " from " << binary_path;
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Exception occurred while loading fused GDN gating kernel: " << e.what();
  }
}

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(const ModelArgs& args,
                                       const QuantArgs& quant_args,
                                       const ParallelArgs& parallel_args,
                                       const torch::TensorOptions& options) {
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  rank_ = parallel_args_.tp_group_->rank();
  tp_size_ = parallel_args.tp_group_->world_size();
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim(); 
  k_size_ = num_k_heads_ * head_k_dim_; 
  v_size_ = num_v_heads_ * head_v_dim_;

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
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));
  // 2. Output projection
  ba_proj_ = register_module("in_proj_ba",
                              ColumnParallelLinear(args.hidden_size(),
                                                   num_v_heads_ * 2,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   options));

  dt_bias_ = register_parameter("dt_bias",
                                torch::ones({num_v_heads_ / tp_size_}, options),
                                /*requires_grad=*/false);
  
  A_log_ = register_parameter("A_log",
                              torch::empty({num_v_heads_ / tp_size_}, options.dtype(torch::kFloat32)),
                              /*requires_grad=*/false);
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
  
  // Load the fused GDN gating kernel
  std::string gdn_triton_kernel_name_ = "fused_gdn_gating_head8_kernel";
  std::string gdn_triton_binary_filename_ = "fused_gdn_gating_head8_kernel.npubin";

  std::string conv_triton_kernel_name_ = "_causal_conv1d_update_kernel_no_cache_len_no_mtp";
  std::string conv_triton_binary_filename_ = "_causal_conv1d_update_kernel_no_cache_len_no_mtp.npubin";

  std::string recurrent_triton_kernel_name_ = "fused_recurrent_gated_delta_rule_fwd_kernel";
  std::string recurrent_triton_binary_filename_ = "fused_recurrent_gated_delta_rule_fwd_kernel.npubin";

  std::string norm_triton_kernel_name_ = "layer_norm_fwd_kernel";
  std::string norm_triton_binary_filename_ = "layer_norm_fwd_kernel.npubin";

  // Load the fused GDN gating kernel
  load_triton_kernel(gdn_triton_kernel_name_, gdn_triton_binary_filename_);
  load_triton_kernel(conv_triton_kernel_name_, conv_triton_binary_filename_);
  load_triton_kernel(recurrent_triton_kernel_name_, recurrent_triton_binary_filename_);
  load_triton_kernel(norm_triton_kernel_name_, norm_triton_binary_filename_)
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
    TORCH_CHECK(t.dim() > 2, "Tensor must have at least 2 dims! but got ", t.dim());
    std::vector<int64_t> new_shape;
    int64_t slice_end = t.dim() - 2;
    auto valid_slice = t.sizes().slice(0, slice_end);
    new_shape = std::vector<int64_t>(valid_slice.begin(), valid_slice.end());
    int64_t last_two_dim = t.size(slice_end) * t.size(slice_end + 1);
    new_shape.push_back(last_two_dim);
    return t.reshape(new_shape);
  };

  q = rearrange_merge(q);
  k = rearrange_merge(k);
  v = rearrange_merge(v);

  torch::Tensor mixed_qkv = torch::cat({q, k, v}, q.dim() - 1);
  mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1,2);
  int64_t seq_len = mixed_qkv.size(2); 
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();

  torch::Tensor g, beta, core_attn_out, last_recurrent_state;
  auto device = mixed_qkv.device();
  auto conv_weight = conv1d_->weight();
  std::cerr << "mixed_qkv - Device: " << mixed_qkv.device()
            << ", Dtype: " << mixed_qkv.dtype()
            << ", Shape: " << mixed_qkv.sizes() << std::endl;
  std::cerr << "conv_weight - Device: " << conv_weight.device()
            << ", Dtype: " << conv_weight.dtype()
            << ", Shape: " << conv_weight.sizes() << std::endl;
  if (attn_metadata.is_prefill) {
    std::cerr << "prefill " << std::endl;
    auto conv_output = torch::conv1d(
        mixed_qkv,
        conv_weight.unsqueeze(1).to(device),
        torch::Tensor(),  
        /*stride=*/std::vector<int64_t>{1},
        /*padding=*/std::vector<int64_t>{3},
        /*dilation=*/std::vector<int64_t>{1},
        /*groups=*/static_cast<int64_t>(mixed_qkv.size(1))
    );
    mixed_qkv = torch::silu(conv_output.slice(2,0,seq_len));
  } else {
    std::cerr << "decode " << std::endl;
    xllm::kernel::CausalConv1dUpdateParams params;
    params.x = mixed_qkv;
    params.conv_state = torch::zeros({1,2048,3}).to(device);
    params.weight = conv_weight;
    params.conv_state_indices = attn_metadata.block_table.slice(1,0,1);

    std::cerr << "conv1d conv_state Shape: " << params.conv_state.sizes() << std::endl;
    if (params.conv_state_indices.has_value()) {
        at::Tensor& conv_state_indices_tensor = params.conv_state_indices.value();
        std::cerr << "conv1d conv_state_indices Shape: " << conv_state_indices_tensor.sizes() << std::endl;
    }
    mixed_qkv = xllm::kernel::causal_conv1d_update(params);
  }

  std::cerr << "after mixed_qkv - Device: " << mixed_qkv.device()
            << ", Dtype: " << mixed_qkv.dtype()
            << ", Shape: " << mixed_qkv.sizes() << std::endl;
  std::cerr << "a_log - Device: " << A_log_.device()
            << ", Dtype: " << A_log_.dtype()
            << ", Shape: " << A_log_.sizes() << std::endl;
  std::cerr << "dt_bias_ - Device: " << dt_bias_.device()
            << ", Dtype: " << dt_bias_.dtype()
            << ", Shape: " << dt_bias_.sizes() << std::endl;

  if (attn_metadata.is_prefill) {
    //g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta = torch::sigmoid(b);
    torch::Tensor A_log_exp = A_log_.exp().to(device);
    torch::Tensor a_float = a.to(torch::kFloat32);
    torch::Tensor a_plus_dt = a_float + dt_bias_.to(device);
    torch::Tensor softplus_out = torch::nn::functional::softplus(
        a_plus_dt,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0)
    );
    g = -A_log_exp * softplus_out;
    g = g.to(a.dtype()).contiguous();
  } else {
    xllm::kernel::FusedGdnGatingParams gdn_params;
    gdn_params.A_log = A_log_.to(device);
    gdn_params.a = a;
    gdn_params.b = b;
    gdn_params.dt_bias = dt_bias_.to(torch::kFloat32).to(device);
    gdn_params.beta = 1.0f;
    gdn_params.threshold = 20.0f;
    std::tie(g, beta) = xllm::kernel::fused_gdn_gating(gdn_params);
  }

  auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
  int64_t repeat_times = num_v_heads_ / num_k_heads_;

  if (repeat_times > 1) {
      processed_q = processed_q.repeat_interleave(repeat_times, 2);
      processed_k = processed_k.repeat_interleave(repeat_times, 2);
  }
  std::cerr << "beta - Device: " << beta.device()
            << ", Dtype: " << beta.dtype()
            << ", Shape: " << beta.sizes() << std::endl;
  std::cerr << "g - Device: " << g.device()
            << ", Dtype: " << g.dtype()
            << ", Shape: " << g.sizes() << std::endl;
  if (attn_metadata.is_prefill) {
    std::tie(core_attn_out, last_recurrent_state) = torch_chunk_gated_delta_rule(processed_q, processed_k, processed_v, g, beta);
  } else {
    xllm::kernel::FusedRecurrentGatedDeltaRuleParams recurrent_gated_params;
    recurrent_gated_params.q = processed_q;
    recurrent_gated_params.k = processed_k;
    recurrent_gated_params.v = processed_v;
    recurrent_gated_params.g = g;
    recurrent_gated_params.beta =std::optional<at::Tensor>(beta);
    recurrent_gated_params.scale = std::optional<float>(1.0f);
    recurrent_gated_params.use_qk_l2norm_in_kernel = true;
    recurrent_gated_params.inplace_final_state = true;
    recurrent_gated_params.initial_state = torch::zeros({1,8,128,128}).to(device);
    recurrent_gated_params.cu_seqlens = attn_metadata.query_start_loc;
    recurrent_gated_params.ssm_state_indices = attn_metadata.block_table.slice(1,0,1);
    recurrent_gated_params.num_accepted_tokens = std::nullopt;
    std::cerr << "processed_q Shape: " << processed_q.sizes() << std::endl;
    std::cerr << "cu_seqlens Shape: " << attn_metadata.query_start_loc.sizes() << std::endl;
    std::cerr << "block_table Shape: " << attn_metadata.block_table.sizes() << std::endl;
    std::tie(core_attn_out, last_recurrent_state) = xllm::kernel::fused_recurrent_gated_delta_rule(recurrent_gated_params);
  }

  auto z_reshaped = z.view({-1, z.size(-1)});
  auto core_attn_out_reshaped = core_attn_out.view({-1, core_attn_out.size(-1)});

  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);

  auto z_shape_og = z.sizes().vec();
  norm_out = norm_out.view(z_shape_og);
  norm_out = norm_out.view({norm_out.size(0), norm_out.size(1), -1});

  auto attn_output = o_proj_->forward(rearrange_merge(norm_out));
  return attn_output; 
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3NextGatedDeltaNetImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) {
  mixed_qkv = mixed_qkv.transpose(1,2);
  int64_t batch_size = mixed_qkv.size(0);
  int64_t seq_len = mixed_qkv.size(1);
  std::vector<int64_t> split_sizes = {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto qkv_split = torch::split(mixed_qkv, split_sizes, 2);

  auto processed_q = qkv_split[0];
  auto processed_k = qkv_split[1];
  auto processed_v = qkv_split[2];

  processed_q = processed_q.view({batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_k = processed_k.view({batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  processed_v = processed_v.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
Qwen3NextGatedDeltaNetImpl::process_qkvz_tensor(const torch::Tensor& qkvz) {
  
  std::vector<int64_t> new_tensor_shape_qkvz = [&]() {
    std::vector<int64_t> dims;
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
  const int64_t rank = rank_;
  const int64_t world_size = tp_size_;
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {k_size_/tp_size_, k_size_/tp_size_, v_size_/tp_size_};
  qkvz_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_qkvz."));
  ba_proj_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_ba."));
  
  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(StateDict({"weight", w.squeeze(1)}),
                             shard_tensor_count, shard_sizes);
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }

  LOAD_SHARDED_WEIGHT(dt_bias, 0);
  LOAD_SHARDED_WEIGHT(A_log, 0);
}

}  // namespace layer
}  // namespace xllm
