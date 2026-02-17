#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#include "vkllm_common.h"
#ifdef __cplusplus
extern "C" {
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
            struct vkllm_pipeline *f32f32f32[4];
            struct vkllm_pipeline *f16f32f32[4];
            struct vkllm_pipeline *f16f16f32[4];
            struct vkllm_pipeline *f16f16f16[4];
        } bin;
        struct
        {
            struct vkllm_pipeline *f32;
            struct vkllm_pipeline *f16;
        } embedding;

        struct
        {
            struct vkllm_pipeline *f16f32f32;
            struct vkllm_pipeline *f16f32f16;
            struct vkllm_pipeline *f32f32f32;
        } rmsnorm;

        struct
        {
            // pipelines array size: a_boardcast_type x b_boardcast_type x transposed_b
            struct vkllm_pipeline *f32f32f32[4][4][2];
            struct vkllm_pipeline *f16f16f16[4][4][2];
            struct vkllm_pipeline *f16f32f16[4][4][2];
        } matmul;

        struct
        {
            struct vkllm_pipeline *f16f16;
            struct vkllm_pipeline *f16f32;
            struct vkllm_pipeline *f32f32;
        } rope;

        struct
        {
            struct vkllm_pipeline *f16f16;
            struct vkllm_pipeline *f16f32;
            struct vkllm_pipeline *f32f32;
        } softmax;

        struct
        {
            struct vkllm_pipeline *f16;
            struct vkllm_pipeline *f32;
        } copy;
    } pipelines;

    struct
    {
        int tensor_alloc_counts;
        int tensor_free_counts;
    } stats;
};

extern vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context);
extern void vkllm_context_free(struct vkllm_context *pcontext);
#ifdef __cplusplus
}
#endif
#endif
