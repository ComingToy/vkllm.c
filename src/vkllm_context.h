#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <log.h>
#include <stdint.h>

#include "vkllm_array.h"
#include "vkllm_dtypes.h"
#include "vkllm_errors.h"

struct vkllm_gpu_device;
struct vkllm_pipeline;

struct vkllm_pipeline_desc
{
    struct vkllm_array_dtype *in_dtypes;
    vkllm_dtype_t dtype;
    struct vkllm_pipeline *pipeline;
};

VKLLM_DEF_ARRAY(pipeline_desc, struct vkllm_pipeline_desc);

struct vkllm_context
{
    const char *appname;
    struct vkllm_gpu_device *device;
    struct
    {
        struct vkllm_array_pipeline_desc *add;
        struct vkllm_array_pipeline_desc *embedding;
    } pipelines;
};

extern vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context);
extern void vkllm_context_free(struct vkllm_context *pcontext);
#ifdef __cplusplus
}
#endif
#endif
