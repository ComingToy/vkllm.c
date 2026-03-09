#include "vkllm_context.h"

#include "src/core/vkllm_hashmap.h"
#include "vkllm_common.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context)
{
    _NEW_AND_CHECK(*context, struct vkllm_context);

    struct vkllm_context *p = *context;
    p->appname = "vkllm";
    p->stats.tensor_alloc_counts = 0;
    p->stats.tensor_free_counts = 0;

    for (int i = 0; i < VKLLM_OP_COUNTS; ++i)
    {
        p->stats.op_time_costs[i] = 0;
    }

    _CHECK(vkllm_hashmap_new(&p->stats.node_time_cost, 1024));
    _CHECK(vkllm_gpu_device_new(p, dev));
    _CHECK(vkllm_create_all_pipelines(p));

    return VKLLM_ERR_OK;
}

void vkllm_context_free(struct vkllm_context *pcontext)
{
    vkllm_hashmap_free(pcontext->stats.node_time_cost);
    vkllm_free_all_pipelines(pcontext);
    vkllm_gpu_device_free(pcontext);
    free(pcontext);
}
