#ifndef __VKLLM_OP_UPDATE_ROWS_H__
#define __VKLLM_OP_UPDATE_ROWS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

struct vkllm_op_update_rows_params
{
    uint32_t offset_rows;
};

extern vkllm_err_t vkllm_op_update_rows_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                             struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_update_rows_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                            struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_update_rows_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                                 struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
