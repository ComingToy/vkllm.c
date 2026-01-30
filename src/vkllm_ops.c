#include "vkllm_ops.h"

const char *vkllm_op_s(const vkllm_op_t op)
{
    static const char *ops[] = {
        "VKLLM_OP_NONE",
        "VKLLM_OP_ADD",
        "VKLLM_OP_EMBEDDING",
        "VKLLM_OP_RMSNORM",
        "VKLLM_OP_MATMUL",
        "VKLLM_OP_ROPE",
    };

    return ops[op];
}
