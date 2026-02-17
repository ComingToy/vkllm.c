#include "vkllm_context.h"

#include "vkllm_common.h"
#include "vkllm_gpu_device.h"

vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context)
{
    _NEW_AND_CHECK(*context, struct vkllm_context);

    struct vkllm_context *p = *context;
    p->appname = "vkllm";
    vkllm_gpu_device_new(p, 0);

    return VKLLM_ERR_OK;
}

void vkllm_context_free(struct vkllm_context *pcontext)
{
    vkllm_gpu_device_free(pcontext);
    free(pcontext);
}
