include(cc_test)

get_filename_component(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(BINARY_DIR "${TEST_DIR}/binary")

cc_test(
  NAME
    fused_gdn_gating_test
  SRCS
    triton_fused_gdn_gating_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

cc_test(
  NAME
    fused_recurrent_gated_delta_rule_test
  SRCS
    triton_recurrent_gated_delta_rule_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

target_compile_definitions(fused_recurrent_gated_delta_rule_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)

target_compile_definitions(fused_gdn_gating_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)
