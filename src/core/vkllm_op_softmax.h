#ifndef __VKLLM_OP_SOFTMAX_H__
#define __VKLLM_OP_SOFTMAX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

struct vkllm_op_softmax_params
{
    int32_t seq_mask;
    uint32_t offsets;
};

extern vkllm_err_t vkllm_op_softmax_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_softmax_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                        struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_softmax_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                             struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
