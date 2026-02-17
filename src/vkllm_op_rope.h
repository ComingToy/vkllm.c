#ifndef __VKLLM_OP_ROPE_H__
#define __VKLLM_OP_ROPE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

struct vkllm_op_rope_params
{
    uint32_t offset; // position offset
    float base;      // base frequency (typically 10000.0)
};

extern vkllm_err_t vkllm_op_rope_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                      struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_rope_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                     struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_rope_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
