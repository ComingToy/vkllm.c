#ifndef __VKLLM_OP_MATMUL_H__
#define __VKLLM_OP_MATMUL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

struct vkllm_op_matmul_params
{
    float scale;
    int32_t act;
    // Explicit layout of srcs[1]: true = B stored [N, K] (row-per-output,
    // e.g. GGUF weights), false = B stored [K, N]. Must be declared by the
    // caller — it cannot always be inferred from shapes (square matrices
    // are ambiguous) and a wrong guess silently selects the wrong pipeline.
    bool transposed_b;
};

// act == 3: dual-weight GEMV (decode FFN). srcs = {x, Wg, Wu}, all f16
// (or all f32); out = [x@Wg | x@Wu] concatenated along the last dim
// (length 2*FF). Requires a single activation row and strictly
// contiguous transposed weights. One dispatch replaces the separate
// gate/up GEMVs; the GLU consumes the halves (bin op 4).
extern vkllm_err_t vkllm_op_mat_mul_vec_dual_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                                  struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_mat_mul_vec_dual_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                                 struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_matmul_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                        struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_matmul_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                       struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_op_matmul_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                            struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
