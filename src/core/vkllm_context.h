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

// Binary op ids: 0 add, 1 sub, 2 mul, 3 div, 4 glu-halves.
#define VKLLM_BIN_OP_COUNTS 5

struct vkllm_context
{
    const char *appname;
    struct vkllm_gpu_device *device;
    struct
    {
        struct
        {
            // ops 0-3: add/sub/mul/div; 4: glu halves (see vkllm_bin_op.comp)
            struct vkllm_pipeline *f32f32f32[VKLLM_BIN_OP_COUNTS];
            struct vkllm_pipeline *f16f32f32[VKLLM_BIN_OP_COUNTS];
            struct vkllm_pipeline *f16f16f32[VKLLM_BIN_OP_COUNTS];
            struct vkllm_pipeline *f16f16f16[VKLLM_BIN_OP_COUNTS];
            struct vkllm_pipeline *f16f32f16[VKLLM_BIN_OP_COUNTS];
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
            // Fused GEMV + residual-add variants (act == 2): same [4][4]
            // boardcast grid, extra residual binding.
            struct vkllm_pipeline *f32f32fa[4][4];
            struct vkllm_pipeline *f16f32fa[4][4];
            struct vkllm_pipeline *f16f16fa[4][4];
        } mat_mul_vec;

        struct
        {
            // Non-transposed GEMV (B row-major [K,N], M == 1): the
            // attention-values shape; one workgroup per (b,c) row.
            struct vkllm_pipeline *f16f32;
            struct vkllm_pipeline *f32f32;
            struct vkllm_pipeline *f16f16;
            // PERMUTE_OUT variants: head-interleaved write (act == 4) —
            // folds the attention-output cont copy into the values GEMV.
            struct vkllm_pipeline *f16f32p;
            struct vkllm_pipeline *f32f32p;
        } mat_mul_vec_nt;

        struct
        {
            // Dual-weight GEMV (gate+up in one dispatch): out = x@Wg | x@Wu.
            struct vkllm_pipeline *f16f32;
            struct vkllm_pipeline *f32f32;
        } mat_mul_vec_dual;

        struct
        {
            // 0: for normal style 1: for neox style
            // f16 variants are f16-storage / fp32-compute: rope's theta math
            // must not run in fp16 (angle precision dies past pos ~1024).
            struct vkllm_pipeline *f16f32[2];
            struct vkllm_pipeline *f32f32[2];
            // FUSED_KV variants: rope writes into the KV cache at a row
            // offset (folds update-rows into the rope dispatch).
            struct vkllm_pipeline *f16f32kv[2];
            struct vkllm_pipeline *f32f32kv[2];
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

        struct
        {
            struct vkllm_pipeline *f16f16;
            struct vkllm_pipeline *f16f32;
            struct vkllm_pipeline *f32f32;
        } arg_max;
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
