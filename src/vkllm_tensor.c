#include "vkllm_tensor.h"
#include "vkllm_dtypes.h"

#include "src/vkllm_common.h"

vkllm_err_t vkllm_calc_strides(const uint32_t *shapes, vkllm_dtype_t dtype, uint32_t *strides)
{
    uint32_t bytes = vkllm_dtype_bytes(dtype);
    strides[3] = bytes;
    strides[2] = shapes[3] * strides[3];
    strides[1] = shapes[2] * strides[2];
    strides[0] = shapes[1] * strides[1];

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_new_tensor(struct vkllm_context *context, const char *name, const uint32_t *shapes,
                             vkllm_dtype_t dtype, struct vkllm_gpu_device *device, vkllm_op_t op,
                             struct vkllm_tensor **srcs, const uint32_t n_srcs, void *params, bool mapped,
                             struct vkllm_tensor **p)
{
    if (!shapes)
    {
        log_error("shape is empty. shapes is NULL");
        return VKLLM_ERR_ARGS;
    }

    for (uint32_t i = 0; i < 4; ++i)
    {
        if (!shapes[i] || shapes[i] == 0)
        {
            log_error("shape is empty. shapes[%u] is 0", i);
            return VKLLM_ERR_ARGS;
        }
    }

    _NEW_AND_CHECK(*p, struct vkllm_tensor);

    struct vkllm_tensor *t = *p;
    t->name = name;
    t->shapes[0] = shapes[0];
    t->shapes[1] = shapes[1];
    t->shapes[2] = shapes[2];
    t->shapes[3] = shapes[3];
    t->dtype = dtype;
    t->device = device;
    t->op = op;

    for (uint32_t i = 0; i < VKLLM_MAX_SRCS; ++i)
    {
        if (i < n_srcs)
        {
            t->srcs[i] = srcs[i];
            continue;
        }
        t->srcs[i] = NULL;
    }

    t->params = params;
    t->mapped = mapped;

    vkllm_calc_strides(shapes, dtype, t->strides);

    return VKLLM_ERR_OK;
}
