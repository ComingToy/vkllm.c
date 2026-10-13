#ifndef __VKLLM_OP_ROPE_H__
#define __VKLLM_OP_ROPE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

struct vkllm_op_rope_params
{
    uint32_t offset; // position offset
    float base;      // base frequency (typically 10000.0)
    bool neox_style;
    // Fused KV write (set by kvcache_bind): the output tensor aliases the
    // KV cache buffer and this dispatch writes cache rows
    // [cache_row, cache_row + seq). Kept in params (NOT srcs) so the graph
    // stays acyclic — kcache->srcs[0] points at this rope for ordering,
    // and a back edge here would loop the recursive traversal.
    bool fused_write;
    uint32_t cache_row;
    // Paired Q+K rope (decode, MHA): this dispatch ALSO rotates the Q rows
    // into q_out's own buffer. srcs = {K_in, Q_in}; q_out->op is set to
    // NONE so it never dispatches on its own (its buffer is written here).
    bool pair_q;
    struct vkllm_tensor *q_out;
};

extern vkllm_err_t vkllm_op_rope_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                      struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_rope_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                     struct vkllm_tensor *tensor);

extern vkllm_err_t vkllm_op_rope_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor);

#ifdef __cplusplus
}
#endif
#endif
