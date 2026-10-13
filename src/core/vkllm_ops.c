#include "vkllm_ops.h"

#include "vkllm_op_arg_max.h"
#include "vkllm_op_bin.h"
#include "vkllm_op_copy.h"
#include "vkllm_op_embedding.h"
#include "vkllm_op_ffn_up_and_gate.h"
#include "vkllm_op_matmul.h"
#include "vkllm_op_rmsnorm.h"
#include "vkllm_op_rope.h"
#include "vkllm_op_softmax.h"
#include "vkllm_op_update_rows.h"

const struct vkllm_op_desc g_op_table[VKLLM_OP_COUNTS] = {
    // VKLLM_OP_NONE: input tensors, nothing to execute.
    [VKLLM_OP_NONE] = {NULL, NULL, NULL},
    [VKLLM_OP_COPY] = {vkllm_op_copy_init, vkllm_op_copy_run, NULL},
    [VKLLM_OP_BIN] = {vkllm_op_bin_init, vkllm_op_bin_run, NULL},
    [VKLLM_OP_EMBEDDING] = {vkllm_op_embedding_init, vkllm_op_embedding_run, NULL},
    [VKLLM_OP_RMSNORM] = {vkllm_op_rmsnorm_init, vkllm_op_rmsnorm_run, NULL},
    [VKLLM_OP_MATMUL] = {vkllm_op_matmul_init, vkllm_op_matmul_run, NULL},
    [VKLLM_OP_ROPE] = {vkllm_op_rope_init, vkllm_op_rope_run, NULL},
    [VKLLM_OP_SOFTMAX] = {vkllm_op_softmax_init, vkllm_op_softmax_run, NULL},
    [VKLLM_OP_FFN_UP_AND_GATE] = {vkllm_op_ffn_up_and_gate_init, vkllm_op_ffn_up_and_gate_run, NULL},
    [VKLLM_OP_UPDATE_ROWS] = {vkllm_op_update_rows_init, vkllm_op_update_rows_run, NULL},
    [VKLLM_OP_ARG_MAX] = {vkllm_op_arg_max_init, vkllm_op_arg_max_run, NULL},
    // VKLLM_OP_REF: borrowed view, sources execute instead.
    [VKLLM_OP_REF] = {NULL, NULL, NULL},
};

const char *vkllm_op_s(const vkllm_op_t op)
{
    static const char *ops[] = {
        "VKLLM_OP_NONE",    "VKLLM_OP_COPY", "VKLLM_OP_BIN",     "VKLLM_OP_EMBEDDING",       "VKLLM_OP_RMSNORM",
        "VKLLM_OP_MATMUL",  "VKLLM_OP_ROPE", "VKLLM_OP_SOFTMAX", "VKLLM_OP_FFN_UP_AND_GATE", "VKLLM_OP_UPDATE_ROWS",
        "VKLLM_OP_ARG_MAX", "VKLLM_OP_REF",
    };

    return ops[op];
}
