include(cc_library)
include(cc_binary)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(PLATFORM "x86_64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(PLATFORM "aarch64-linux")
endif()


set(TORCH_API_SRCS
  torch_api/npu_triton_fused_gdn_gating.cpp
  torch_api/npu_triton_recurrent_gated_delta_rule.cpp
  torch_api/npu_triton_causal_conv1d_update.cpp
  torch_api/npu_triton_layer_norm_fwd.cpp
)

cc_library(
  NAME
    triton_adapter
  HDRS
    kernel_handle.h
    kernel_registry.h
    kernel_loader.h
    macros.h
    kernel_launchers.h
    torch_api/triton_ops_api.h
    torch_api/utils.h
  INCLUDES
    ${ASCEND_HOME_PATH}/${PLATFORM}/include/experiment/platform
    ${ASCEND_HOME_PATH}/${PLATFORM}/include/experiment/msprof
  LINKOPTS
    "$ENV{ASCEND_HOME_PATH}/lib64/libruntime.so"
  SRCS
    kernel_handle.cpp
    kernel_registry.cpp
    kernel_loader.cpp
    ${TORCH_API_SRCS}
  DEPS
    ascendcl
    nlohmann_json::nlohmann_json
    glog::glog
)

add_subdirectory(test)
