#ifndef __VKLLM_OP_EMBEDDING_H__
#define __VKLLM_OP_EMBEDDING_H__

#include "vkllm_errors.h"

struct vkllm_context;
struct vkllm_commands;
struct vkllm_tensor;
extern vkllm_err_t vkllm_op_embedding(struct vkllm_context *context, struct vkllm_commands *commands,
                                      struct vkllm_tensor *tensor);

#endif
