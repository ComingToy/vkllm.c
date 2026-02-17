#ifndef __VKLLM_ADD_H__
#define __VKLLM_ADD_H__

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

extern vkllm_err_t vkllm_op_add(struct vkllm_context *context, struct vkllm_commands *commands,
                                struct vkllm_tensor *tensor);

#endif
