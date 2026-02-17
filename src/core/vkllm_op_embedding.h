#ifndef __VKLLM_OP_EMBEDDING_H__
#define __VKLLM_OP_EMBEDDING_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_errors.h"

struct vkllm_context;
struct vkllm_commands;
struct vkllm_tensor;

extern vkllm_err_t vkllm_op_embedding_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                           struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_embedding_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_embedding_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                               struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
