/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include "triton_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void validate_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " tensor is not defined");
  TORCH_CHECK(tensor.is_contiguous(), name, " tensor must be contiguous");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name,
              " tensor must be on NPU device");
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> npu_fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta,
    float threshold) {
  validate_tensor(A_log, "A_log");
  validate_tensor(a, "a");
  validate_tensor(b, "b");
  validate_tensor(dt_bias, "dt_bias");

  TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
  TORCH_CHECK(a.dtype() == torch::kFloat16, "a must be float16");
  TORCH_CHECK(b.dtype() == torch::kFloat16, "b must be float16");
  TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");

  TORCH_CHECK(A_log.dim() == 1, "A_log must be 1D tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D tensor (batch, num_heads)");
  TORCH_CHECK(b.dim() == 2, "b must be 2D tensor (batch, num_heads)");
  TORCH_CHECK(dt_bias.dim() == 1, "dt_bias must be 1D tensor");

  int64_t batch = a.size(0);
  int64_t num_heads = a.size(1);

  TORCH_CHECK(A_log.size(0) == num_heads,
              "A_log size must match num_heads: ",
              A_log.size(0),
              " != ",
              num_heads);
  TORCH_CHECK(b.size(0) == batch && b.size(1) == num_heads,
              "b shape must match a shape");
  TORCH_CHECK(dt_bias.size(0) == num_heads,
              "dt_bias size must match num_heads: ",
              dt_bias.size(0),
              " != ",
              num_heads);

  torch::Tensor g = torch::empty(
      {1, batch, num_heads},
      torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

  torch::Tensor beta_output =
      torch::empty({1, batch, num_heads},
                   torch::TensorOptions().dtype(b.dtype()).device(b.device()));

  int32_t seq_len = 1;
  int32_t gridX = static_cast<int32_t>(batch);
  int32_t gridY = seq_len;
  int32_t gridZ =
      static_cast<int32_t>((num_heads + 7) / 8);  // ceil(num_heads / 8)

  auto npuStream = c10_npu::getCurrentNPUStream();
  rtStream_t stream = static_cast<rtStream_t>(npuStream.stream());

  void* gPtr = g.data_ptr();
  void* betaOutputPtr = beta_output.data_ptr();
  void* ALogPtr = const_cast<void*>(A_log.data_ptr());
  void* aPtr = const_cast<void*>(a.data_ptr());
  void* bPtr = const_cast<void*>(b.data_ptr());
  void* dtBiasPtr = const_cast<void*>(dt_bias.data_ptr());
  void* workspace_addr = nullptr;
  void* sync_block_lock = nullptr;

  auto ret = launchers::fused_gdn_gating_head8_kernel(
      stream,
      gridX,
      gridY,
      gridZ,
      workspace_addr,
      sync_block_lock,
      gPtr,
      betaOutputPtr,
      ALogPtr,
      aPtr,
      bPtr,
      dtBiasPtr,
      seq_len);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "Failed to launch kernel "
    << "fused_gdn_gating_head8_kernel" << " : error=" << ret;
  }
  return std::make_pair(g, beta_output);
}

}  // namespace xllm::kernel::npu


