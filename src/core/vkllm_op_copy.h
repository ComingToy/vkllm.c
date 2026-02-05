#ifndef __VKLLM_OP_COPY_H__
#define __VKLLM_OP_COPY_H__

#include "vkllm_errors.h"

struct vkllm_context;
struct vkllm_tensor;
struct vkllm_commands;
extern vkllm_err_t vkllm_op_copy_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                      struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_copy_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                     struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_copy_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor);

#endif
