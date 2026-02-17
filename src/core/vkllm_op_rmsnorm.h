#ifndef __VKLLM_OP_RMSNORM_H__
#define __VKLLM_OP_RMSNORM_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_errors.h"

struct vkllm_context;
struct vkllm_commands;
struct vkllm_tensor;

struct vkllm_op_rmsnorm_params
{
    float power;
    float eps;
};
extern vkllm_err_t vkllm_op_rmsnorm_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_rmsnorm_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                        struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_rmsnorm_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                             struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
