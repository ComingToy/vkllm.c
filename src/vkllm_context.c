#include "vkllm_context.h"

#include "vkllm_common.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context)
{
    _NEW_AND_CHECK(*context, struct vkllm_context);

    struct vkllm_context *p = *context;
    p->appname = "vkllm";

    _CHECK(vkllm_array_pipeline_desc_new(&p->pipelines.add, 32 * sizeof(struct vkllm_pipeline_desc)));
    _CHECK(vkllm_array_pipeline_desc_new(&p->pipelines.embedding, 32 * sizeof(struct vkllm_pipeline_desc)));
    _CHECK(vkllm_gpu_device_new(p, dev));
    _CHECK(vkllm_create_all_pipelines(p));

    return VKLLM_ERR_OK;
}

void vkllm_context_free(struct vkllm_context *pcontext)
{
    vkllm_free_all_pipelines(pcontext);
    vkllm_array_pipeline_desc_free(pcontext->pipelines.add);
    vkllm_gpu_device_free(pcontext);
    free(pcontext);
}
