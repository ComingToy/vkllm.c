#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#include "vkllm_common.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <log.h>
#include <stdint.h>

#include "vkllm_errors.h"
#include "vkllm_ops.h"

struct vkllm_gpu_device;
struct vkllm_pipeline;
struct vkllm_hashmap;

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
            struct vkllm_pipeline *f16f32f16[4];
        } bin;
        struct
        {
            struct vkllm_pipeline *f32;
            struct vkllm_pipeline *f16;
        } embedding;

        struct
        {
            struct vkllm_pipeline *f16f32f16;
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
            // pipeline array size: a_boardcast_type x b_boardcast_type
            struct vkllm_pipeline *f32f32[4][4];
            struct vkllm_pipeline *f16f32[4][4];
            struct vkllm_pipeline *f16f16[4][4];
        } mat_mul_vec;

        struct
        {
            // 0: for normal style 1: for neox style
            struct vkllm_pipeline *f16f16[2];
            struct vkllm_pipeline *f16f32[2];
            struct vkllm_pipeline *f32f32[2];
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

        struct
        {
            struct vkllm_pipeline *f16;
            struct vkllm_pipeline *f32;
        } update_rows;

        struct
        {
            struct vkllm_pipeline *f16f32f16;
            struct vkllm_pipeline *f16f32f32;
            struct vkllm_pipeline *f32f32f32;
        } ffn;
    } pipelines;

    struct
    {
        int tensor_alloc_counts;
        int tensor_free_counts;
        uint64_t op_time_costs[(int)VKLLM_OP_COUNTS];
        struct vkllm_hashmap *node_time_cost;
    } stats;
};

extern vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context);
extern void vkllm_context_free(struct vkllm_context *pcontext);
#ifdef __cplusplus
}
#endif
#endif
