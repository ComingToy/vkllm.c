#include "vkllm_tensor.h"

#include "src/vkllm_common.h"

vkllm_err_t vkllm_new_tensor(struct vkllm_context *context, const char *name, const uint32_t *shapes,
                             vkllm_dtype_t dtype, struct vkllm_tensor **srcs, const uint32_t n_srcs, void *params,
                             bool mapped, struct vkllm_tensor **p)
{
    if (!shapes)
    {
        log_error("shape is empty. shapes is NULL");
        return VKLLM_ERR_ARGS;
    }

    _NEW_AND_CHECK(*p, struct vkllm_tensor);

    struct vkllm_tensor *t = *p;
    t->name = name;
    t->shapes[0] = shapes[0];
    t->shapes[1] = shapes[1];
    t->shapes[2] = shapes[2];
    t->shapes[3] = shapes[3];
    t->dtype = dtype;

    for (uint32_t i = 0; i < VKLLM_MAX_SRCS; ++i)
    {
        if (i < n_srcs)
        {
            t->srcs[i] = srcs[i];
            continue;
        }
        t->srcs[i] = NULL;
    }

    return VKLLM_ERR_OK;
}
