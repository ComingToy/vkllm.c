#ifndef __VKLLM_OP_BIN__
#define __VKLLM_OP_BIN__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

extern vkllm_err_t vkllm_op_bin_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                        struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_bin_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                    struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_bin_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
