#ifndef __VKLLM_OPS_H__
#define __VKLLM_OPS_H__

typedef enum
{
    VKLLM_OP_NONE = 0,
    VKLLM_OP_ADD,
    VKLLM_OP_EMBEDDING,
} vkllm_op_t;

extern const char *vkllm_op_s(const vkllm_op_t op);
#endif
