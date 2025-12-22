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

#pragma once

#include "macros.h"

namespace xllm::kernel::npu {
namespace launchers {
 
#define FUSED_GDN_GATING_ARG_LIST(OP) \
    OP(void*, g)                        \
    OP(void*, beta_output)              \
    OP(void*, A_log)                    \
    OP(void*, a)                        \
    OP(void*, b)                        \
    OP(void*, dt_bias)                  \
    OP(int32_t, seq_len)
 
REG_KERNEL_ARGS(fused_gdn_gating_head8_kernel, FUSED_GDN_GATING_ARG_LIST)
REG_KERNEL_LAUNCHER(fused_gdn_gating_head8_kernel, FUSED_GDN_GATING_ARG_LIST)
 
#define FUSED_RECURRENT_GATED_DELTA_RULE_ARG_LIST(OP) \
    OP(void*, q)                                        \
    OP(void*, k)                                        \
    OP(void*, v)                                        \
    OP(void*, g)                                        \
    OP(void*, beta)                                     \
    OP(void*, o)                                        \
    OP(void*, h0)                                       \
    OP(void*, ht)                                       \
    OP(void*, cu_seqlens)                               \
    OP(float, scale)                                    \
    OP(int32_t, N)                                      \
    OP(int32_t, T)
 
REG_KERNEL_ARGS(fused_recurrent_gated_delta_rule_fwd_kernel,
                FUSED_RECURRENT_GATED_DELTA_RULE_ARG_LIST)
REG_KERNEL_LAUNCHER(fused_recurrent_gated_delta_rule_fwd_kernel,
                    FUSED_RECURRENT_GATED_DELTA_RULE_ARG_LIST)
 
// no input bias, cache_lines and num_accepted_tokens
#define CAUSAL_CONV1D_UPDATE_NO_CACHE_NO_MTP_ARG_LIST(OP) \
    OP(void*, x)                                            \
    OP(void*, conv_state)                                   \
    OP(void*, weight)                                      \
    OP(void*, conv_state_indices)                          \
    OP(void*, out)                                         \
    OP(int32_t, pad_slot_id)                               
 
REG_KERNEL_ARGS(_causal_conv1d_update_kernel_no_cache_len_no_mtp,
                CAUSAL_CONV1D_UPDATE_NO_CACHE_NO_MTP_ARG_LIST)
REG_KERNEL_LAUNCHER(_causal_conv1d_update_kernel_no_cache_len_no_mtp,
                    CAUSAL_CONV1D_UPDATE_NO_CACHE_NO_MTP_ARG_LIST)
}  // namespace launchers
}  // namespace xllm::kernel::npu
