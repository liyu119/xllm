#include "npu_triton_layer_norm_fwd.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cmath>
#include <limits>
#include <algorithm>  // 补充std::clamp必需的头文件
#include <iostream>   // 补充std::cout必需的头文件
#include "experiment/runtime/runtime/rt.h"
#include "kernel_launchers.h"
#include "utils.h"

namespace xllm::kernel::npu {

constexpr int64_t MAX_CORES = 65535;  // Triton默认最大核心数
constexpr int64_t MAX_FUSED_BYTES = 65536;  // 64KB

inline int64_t next_power_of_2(int64_t n) {
    if (n <= 1) return 1;
    // 修复：n-1为0时__builtin_clzll崩溃问题，不改变原有逻辑
    uint64_t val = static_cast<uint64_t>(n - 1);
    return 1LL << (64 - __builtin_clzll(val ? val : 1));
}

torch::Tensor layer_norm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps,
    const c10::optional<torch::Tensor>& z,
    int64_t group_size,
    bool norm_before_gate,
    bool is_rms_norm
) {
    std::cout << "-----------------layer_norm_fwd start----------------" <<std::endl;
    c10::IntArrayRef x_shape_og = x.sizes();
    int64_t last_dim = x.size(-1);
    torch::Tensor x_2d = x.reshape({-1, last_dim});

    // 步长检查（确保最后一维是连续的）
    TORCH_CHECK(x_2d.stride(-1) == 1, "x stride(-1) must be 1");

    // 基础维度检查
    TORCH_CHECK(x_2d.dim() == 2, "x must be 2-dimensional (M, N)");
    const auto M = x_2d.size(0);
    const auto N = x_2d.size(1);


    // 处理group_size - 修复：重命名gs为group_size_val避免命名冲突
    const int64_t group_size_val = group_size;
    TORCH_CHECK(N % group_size_val == 0, "N must be divisible by group_size");
    const int64_t ngroups = N / group_size_val;

    // z张量检查
    torch::Tensor z_2d;
    if (z.has_value()) {
        TORCH_CHECK(z->stride(-1) == 1, "z stride(-1) must be 1");
        TORCH_CHECK(z->sizes() == x.sizes(), "z shape must match x");
        z_2d = z->reshape({-1, last_dim});
    }

    // weight检查
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == N, 
                "weight must be 1-dimensional with size N");
    TORCH_CHECK(weight.stride(-1) == 1, "weight stride(-1) must be 1");

    // bias检查 - 修复：bias是普通Tensor，替换has_value()和->
    if (bias.defined()) {
        TORCH_CHECK(bias.stride(-1) == 1, "bias stride(-1) must be 1");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, 
                    "bias must be 1-dimensional with size N");
    }

    // 输出张量分配
    torch::Tensor out_tensor = torch::empty_like(x);
    TORCH_CHECK(out_tensor.sizes() == x.sizes(), "out shape must match x");
    TORCH_CHECK(out_tensor.stride(-1) == 1, "out stride(-1) must be 1");

    // 均值和逆标准差张量分配
    torch::Tensor mean, rstd;
    if (!is_rms_norm) {
        mean = torch::empty({ngroups * M}, torch::dtype(torch::kFloat32).device(x.device()));
    }
    rstd = torch::empty({ngroups * M}, torch::dtype(torch::kFloat32).device(x.device()));

    // 计算块大小（BLOCK_N） - 修复：替换gs为group_size_val
    const int64_t elem_size = x.element_size();
    const int64_t MAX_FUSED_SIZE = MAX_FUSED_BYTES / elem_size;
    const int64_t BLOCK_N = std::min(MAX_FUSED_SIZE, next_power_of_2(group_size_val));

    // 检查group_size限制 - 修复：替换gs为group_size_val
    TORCH_CHECK(group_size_val <= BLOCK_N, 
                "This layer norm doesn't support feature dim >= 64KB.");

    // 计算warp数量（num_warps） - 修复：std::clamp类型不匹配
    const int64_t warp_base = BLOCK_N / 256;
    const int64_t num_warps = std::clamp<int64_t>(warp_base, 1, 8);

    // 计算grid维度
    const int64_t grid_m = std::min(M, MAX_CORES);
    const auto grid = std::make_pair(grid_m, ngroups);

    std::cout << "-----------------layer_norm_fwd end----------------" <<std::endl;
    
    

    return out_tensor;
}
}
