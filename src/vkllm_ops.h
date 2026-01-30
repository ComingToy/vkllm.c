#ifndef __VKLLM_OPS_H__
#define __VKLLM_OPS_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    VKLLM_OP_NONE = 0,
    VKLLM_OP_ADD,
    VKLLM_OP_EMBEDDING,
    VKLLM_OP_RMSNORM,
    VKLLM_OP_MATMUL,
    VKLLM_OP_ROPE,
} vkllm_op_t;

extern const char *vkllm_op_s(const vkllm_op_t op);

#ifdef __cplusplus
}
#endif
#endif
