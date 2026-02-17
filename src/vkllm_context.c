#include "vkllm_context.h"

#include "vkllm_common.h"

vkllm_err_t vkllm_new_context(struct vkllm_context **context)
{
    _NEW_AND_CHECK(*context, struct vkllm_context);

    struct vkllm_context *p = *context;
    p->appname = "vkllm";

    return VKLLM_ERR_OK;
}

void vkllm_destroy_context(struct vkllm_context *pcontext)
{
    free(pcontext);
}
