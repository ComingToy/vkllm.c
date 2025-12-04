#include "vkllm_tensor.h"

#include "src/vkllm_common.h"

vkllm_err_t vkllm_new_tensor(struct vkllm_context *context, const uint32_t *shapes, const uint32_t n_shape,
                             vkllm_dtype_t dtype, struct vkllm_tensor *srcs, const uint32_t n_srcs, void *params,
                             bool mapped, struct vkllm_tensor **p)
{
    if (!shapes || !n_shape)
    {
        log_error("shape is empty. shapes is NULL: %s, n_shape: %u", shapes ? "false" : "true", n_shape);
        return VKLLM_ERR_ARGS;
    }

    _NEW_AND_CHECK(*p, struct vkllm_tensor);
    return VKLLM_ERR_OK;
}
