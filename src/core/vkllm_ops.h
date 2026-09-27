#ifndef __VKLLM_OPS_H__
#define __VKLLM_OPS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_errors.h"

struct vkllm_context;
struct vkllm_commands;
struct vkllm_tensor;

typedef enum
{
    VKLLM_OP_NONE = 0,
    VKLLM_OP_COPY,
    VKLLM_OP_BIN,
    VKLLM_OP_EMBEDDING,
    VKLLM_OP_RMSNORM,
    VKLLM_OP_MATMUL,
    VKLLM_OP_ROPE,
    VKLLM_OP_SOFTMAX,
    VKLLM_OP_FFN_UP_AND_GATE,
    VKLLM_OP_UPDATE_ROWS,
    VKLLM_OP_ARG_MAX,
    VKLLM_OP_REF,
    VKLLM_OP_COUNTS,
} vkllm_op_t;

// Op vtable: every op registers its init/run/post_run here. graph.c walks
// the graph and dispatches through this table — adding an op is one enum
// entry plus one row in g_op_table (vkllm_ops.c), not a new case in each
// of graph.c's three switches.
struct vkllm_op_desc
{
    // NULL means "nothing to do for this phase" (input/none/ref tensors).
    vkllm_err_t (*init)(struct vkllm_context *context, struct vkllm_commands *commands, struct vkllm_tensor *tensor);
    vkllm_err_t (*run)(struct vkllm_context *context, struct vkllm_commands *commands, struct vkllm_tensor *tensor);
    vkllm_err_t (*post_run)(struct vkllm_context *context, struct vkllm_commands *commands,
                            struct vkllm_tensor *tensor);
};

// Dispatch table indexed by vkllm_op_t; entries for VKLLM_OP_NONE and
// VKLLM_OP_REF are all-NULL.
extern const struct vkllm_op_desc g_op_table[VKLLM_OP_COUNTS];

extern const char *vkllm_op_s(const vkllm_op_t op);

#ifdef __cplusplus
}
#endif
#endif
