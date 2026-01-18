#!/bin/sh

VKLLM_BUILD_DIR=`pwd`
VKLLM_BINARY_DIR="${VKLLM_BUILD_DIR}/bazel-bin/tests"

${VKLLM_BINARY_DIR}/vkllm_test_transfer
${VKLLM_BINARY_DIR}/vkllm_test_op_add
${VKLLM_BINARY_DIR}/vkllm_test_op_embedding
${VKLLM_BINARY_DIR}/vkllm_test_op_matmul
${VKLLM_BINARY_DIR}/vkllm_test_op_permute
${VKLLM_BINARY_DIR}/vkllm_test_op_rmsnorm
