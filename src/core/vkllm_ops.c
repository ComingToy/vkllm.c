#include "vkllm_ops.h"

const char *vkllm_op_s(const vkllm_op_t op)
{
    static const char *ops[] = {
        "VKLLM_OP_NONE",
        "VKLLM_OP_COPY",
        "VKLLM_OP_BIN",
        "VKLLM_OP_EMBEDDING",
        "VKLLM_OP_RMSNORM",
        "VKLLM_OP_MATMUL",
        "VKLLM_OP_ROPE",
        "VKLLM_OP_SOFTMAX",
    };

    return ops[op];
}
