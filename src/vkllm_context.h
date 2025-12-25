#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#include "src/vkllm_common.h"
#ifdef __cplusplus
extern "C"
{
#endif

#include <log.h>
#include <stdint.h>

#include "vkllm_errors.h"

    struct vkllm_gpu_device;
    struct vkllm_pipeline;

    struct vkllm_context
    {
        const char *appname;
        struct vkllm_gpu_device *device;
        struct
        {
            struct
            {
                struct vkllm_pipeline *pipeline_f32f32f32;
                struct vkllm_pipeline *pipeline_f16f32f32;
                struct vkllm_pipeline *pipeline_f16f16f32;
                struct vkllm_pipeline *pipeline_f16f16f16;
                struct vkllm_pipeline *pipeline_f16f32f16;
            } add;
            struct
            {
                struct vkllm_pipeline *f32;
                struct vkllm_pipeline *f16;
            } embedding;
        } pipelines;
    };

    extern vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context);
    extern void vkllm_context_free(struct vkllm_context *pcontext);
#ifdef __cplusplus
}
#endif
#endif
