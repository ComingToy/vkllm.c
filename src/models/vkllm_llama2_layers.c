#include "vkllm_llama2_layers.h"
#include "../core/vkllm_context.h"
#include "../core/vkllm_graph.h"
#include "../core/vkllm_op_matmul.h"
#include "../core/vkllm_op_rmsnorm.h"
#include "../core/vkllm_op_softmax.h"
#include "../core/vkllm_tensor.h"
#include "src/core/vkllm_common.h"
#include <math.h>

vkllm_err_t vkllm_llama2_build_ffn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                         struct vkllm_tensor *input, struct vkllm_llama2_ffn_layer_params params,
                                         const char *name)
{
    _CHECK_ARGS(context && graph && input && params.WD && params.WG && params.WU);
    struct vkllm_tensor *norm = NULL, *up = NULL, *gate = NULL, *down = NULL, *gate_mul = NULL;
    vkllm_err_t err = VKLLM_ERR_OK;
    char scope_buf[128];

    uint32_t batch = input->shapes[0];
    uint32_t channel = input->shapes[1];
    uint32_t seq_len = input->shapes[2];
    uint32_t up_dim = params.WU->shapes[2];

    struct vkllm_tensor *norm_srcs[] = {input, params.norm_weight};
    struct vkllm_op_rmsnorm_params norm_params = {.power = params.norm_power, .eps = params.norm_eps};
    snprintf(scope_buf, sizeof(scope_buf), "%s.norm", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, input->shapes, input->dtype, VKLLM_OP_RMSNORM, norm_srcs, 2,
                            &norm_params, sizeof(norm_params), false, &norm));
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, norm), err, fail_free_norm);

    struct vkllm_tensor *up_srcs[] = {norm, params.WU};
    uint32_t ffn_up_shapes[] = {batch, channel, seq_len, up_dim};
    struct vkllm_op_matmul_params up_matmul_params = {.scale = 1.0, .act = 0};

    snprintf(scope_buf, sizeof(scope_buf), "%s.up", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, norm->dtype, VKLLM_OP_MATMUL, up_srcs, 2,
                                 &up_matmul_params, sizeof(up_matmul_params), false, &up),
                err, fail_free_norm);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, up), err, fail_free_up);

    struct vkllm_tensor *gate_srcs[] = {norm, params.WG};
    struct vkllm_op_matmul_params gate_matmul_params = {.scale = 1.0, .act = 1};
    snprintf(scope_buf, sizeof(scope_buf), "%s.gate", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, norm->dtype, VKLLM_OP_MATMUL, gate_srcs, 2,
                                 &gate_matmul_params, sizeof(gate_matmul_params), false, &gate),
                err, fail_free_up);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, gate), err, fail_free_gate);

    struct vkllm_tensor *gate_mul_srcs[] = {up, gate};
    int32_t bin_op = 2; // times
    snprintf(scope_buf, sizeof(scope_buf), "%s.mul", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, gate->dtype, VKLLM_OP_BIN, gate_mul_srcs, 2,
                                 &bin_op, sizeof(bin_op), false, &gate_mul),
                err, fail_free_gate);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, gate_mul), err, fail_free_gate_mul);

    struct vkllm_tensor *down_srcs[] = {gate_mul, params.WD};
    struct vkllm_op_matmul_params down_params = {.scale = 1.0, .act = 0};
    uint32_t ffn_down_shapes[] = {batch, channel, seq_len, params.WD->shapes[2]};
    snprintf(scope_buf, sizeof(scope_buf), "%s.down", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, ffn_down_shapes, gate_mul->dtype, VKLLM_OP_MATMUL, down_srcs, 2,
                                 &down_params, sizeof(down_params), false, &down),
                err, fail_free_down);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, down), err, fail_free_down);

    struct vkllm_tensor *output_srcs[] = {input, down};
    int32_t add_op = 0;
    struct vkllm_tensor *output;

    snprintf(scope_buf, sizeof(scope_buf), "%s.add", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, input->shapes, input->dtype, VKLLM_OP_BIN, output_srcs, 2, &add_op,
                                 sizeof(add_op), false, &output),
                err, fail_free_down);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, output), err, fail_free_output);

    return VKLLM_ERR_OK;

fail_free_output:
    vkllm_tensor_free(context, output);
fail_free_down:
    vkllm_tensor_free(context, down);
fail_free_gate_mul:
    vkllm_tensor_free(context, gate_mul);
fail_free_gate:
    vkllm_tensor_free(context, gate);
fail_free_up:
    vkllm_tensor_free(context, up);
fail_free_norm:
    vkllm_tensor_free(context, norm);
    return err;
}

vkllm_err_t vkllm_llama2_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                               struct vkllm_tensor *input,
                                               struct vkllm_llama2_self_attn_layer_params params, const char *name)
{
    _CHECK_ARGS(context && graph && input && params.WQ && params.WK && params.WV);

    struct vkllm_op_rope_params rope_params = {.base = params.freq_base, .offset = params.offsets};

    uint32_t hidden_dim = params.WQ->shapes[3];              // hidden_dim from weight matrix
    uint32_t num_head_times_head_dim = params.WQ->shapes[2]; // num_head * head_dim
    uint32_t head_dim = num_head_times_head_dim / params.num_head;

    // Validate weight shapes
    _CHECK_ARGS(params.WK->shapes[0] == 1 && params.WK->shapes[1] == 1);
    _CHECK_ARGS(params.WV->shapes[0] == 1 && params.WV->shapes[1] == 1);
    _CHECK_ARGS(params.WK->shapes[2] == num_head_times_head_dim && params.WK->shapes[3] == hidden_dim);
    _CHECK_ARGS(params.WV->shapes[2] == num_head_times_head_dim && params.WV->shapes[3] == hidden_dim);

    // Get input dimensions
    uint32_t batch = input->shapes[0];
    uint32_t seq_len = input->shapes[2];

    // Validate input shape
    _CHECK_ARGS(input->shapes[3] == hidden_dim);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_tensor *norm = NULL;
    struct vkllm_tensor *Q = NULL, *K = NULL, *V = NULL;
    struct vkllm_tensor *RQ = NULL, *RK = NULL;
    struct vkllm_tensor *scores = NULL, *attn_weights = NULL, *output = NULL;
    char scope_buf[128];

    struct vkllm_tensor *norm_srcs[] = {input, params.norm_weight};
    struct vkllm_op_rmsnorm_params norm_params = {.power = params.norm_power, .eps = params.norm_eps};
    snprintf(scope_buf, sizeof(scope_buf), "%s.norm", name);

    _CHECK(vkllm_tensor_new(context, scope_buf, input->shapes, input->dtype, VKLLM_OP_RMSNORM, norm_srcs, 2,
                            &norm_params, sizeof(norm_params), false, &norm));
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, norm), err, fail_free_norm);

    // Step 1: Compute Q = input @ WQ^T
    // Q shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t Q_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *Q_srcs[] = {norm, params.WQ};
    struct vkllm_op_matmul_params matmul_params = {.act = 0, .scale = 1.0f};
    snprintf(scope_buf, sizeof(scope_buf), "%s.Q", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, Q_shapes, norm->dtype, VKLLM_OP_MATMUL, Q_srcs, 2, &matmul_params,
                                 sizeof(matmul_params), false, &Q),
                err, fail_free_norm);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, Q), err, fail_free_Q);

    // Step 2: Compute K = input @ WK^T
    // K shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t K_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *K_srcs[] = {norm, params.WK};
    snprintf(scope_buf, sizeof(scope_buf), "%s.K", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, K_shapes, norm->dtype, VKLLM_OP_MATMUL, K_srcs, 2, &matmul_params,
                                 sizeof(matmul_params), false, &K),
                err, fail_free_Q);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, K), err, fail_free_K);

    // Step 3: Compute V = input @ WV^T
    // V shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t V_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *V_srcs[] = {norm, params.WV};
    snprintf(scope_buf, sizeof(scope_buf), "%s.V", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, V_shapes, norm->dtype, VKLLM_OP_MATMUL, V_srcs, 2, &matmul_params,
                                 sizeof(matmul_params), false, &V),
                err, fail_free_K);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, V), err, fail_free_V);

    // Step 4: Reshape Q, K, V to split heads
    // First reshape Q from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t Q_reshaped_shapes[4] = {batch, seq_len, params.num_head, head_dim};
    struct vkllm_tensor *Q_ref = NULL;
    _CHECK_JUMP(vkllm_tensor_copy_ref(context, Q, &Q_ref), err, fail_free_V);
    _CHECK_JUMP(vkllm_tensor_reshape(context, Q_ref, Q_reshaped_shapes), err, fail_free_Q_ref);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, Q_ref), err, fail_free_Q_ref);

    // Then permute Q to [batch, num_head, seq_len, head_dim]
    uint32_t Q_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    _CHECK_JUMP(vkllm_tensor_permute(context, Q_ref, Q_permute_axis), err, fail_free_Q_ref);

    // First reshape K from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t K_reshaped_shapes[4] = {batch, seq_len, params.num_head, head_dim};
    struct vkllm_tensor *K_ref = NULL;
    _CHECK_JUMP(vkllm_tensor_copy_ref(context, K, &K_ref), err, fail_free_Q_ref);
    _CHECK_JUMP(vkllm_tensor_reshape(context, K_ref, K_reshaped_shapes), err, fail_free_K_ref);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, K_ref), err, fail_free_K_ref);

    // Then permute K to [batch, num_head, seq_len, head_dim]
    uint32_t K_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    _CHECK_JUMP(vkllm_tensor_permute(context, K_ref, K_permute_axis), err, fail_free_K_ref);

    // First reshape V from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t V_reshaped_shapes[4] = {batch, seq_len, params.num_head, head_dim};
    struct vkllm_tensor *V_ref = NULL;
    _CHECK_JUMP(vkllm_tensor_copy_ref(context, V, &V_ref), err, fail_free_K_ref);
    _CHECK_JUMP(vkllm_tensor_reshape(context, V_ref, V_reshaped_shapes), err, fail_free_V_ref);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, V_ref), err, fail_free_V_ref);

    // Then permute V to [batch, num_head, seq_len, head_dim]
    uint32_t V_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    _CHECK_JUMP(vkllm_tensor_permute(context, V_ref, V_permute_axis), err, fail_free_V_ref);

    snprintf(scope_buf, sizeof(scope_buf), "%s.RQ", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, Q_ref->shapes, Q_ref->dtype, VKLLM_OP_ROPE, &Q_ref, 1,
                                 &rope_params, sizeof(rope_params), false, &RQ),
                err, fail_free_V_ref);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, RQ), err, fail_free_RQ);

    snprintf(scope_buf, sizeof(scope_buf), "%s.RK", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, K_ref->shapes, K_ref->dtype, VKLLM_OP_ROPE, &K_ref, 1,
                                 &rope_params, sizeof(rope_params), false, &RK),
                err, fail_free_RQ);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, RK), err, fail_free_RK);

    // Step 6: Compute scores = Q @ K^T / sqrt(head_dim)
    // scores shape: [batch, num_head, seq_len, seq_len]
    uint32_t scores_shapes[4] = {batch, params.num_head, seq_len, seq_len};
    struct vkllm_tensor *scores_srcs[] = {RQ, RK};
    matmul_params.scale = 1.0f / sqrtf((float)head_dim);
    snprintf(scope_buf, sizeof(scope_buf), "%s.scores", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, scores_shapes, input->dtype, VKLLM_OP_MATMUL, scores_srcs, 2,
                                 &matmul_params, sizeof(matmul_params), false, &scores),
                err, fail_free_RK);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, scores), err, fail_free_scores);

    // Step 7: Apply softmax to get attention weights
    // attn_weights shape: [batch, num_head, seq_len, seq_len]
    struct vkllm_tensor *softmax_srcs[] = {scores};
    struct vkllm_op_softmax_params softmax_params = {.seq_mask = 1, // No masking
                                                     .offsets = rope_params.offset};

    snprintf(scope_buf, sizeof(scope_buf), "%s.attn_weights", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, scores_shapes, input->dtype, VKLLM_OP_SOFTMAX, softmax_srcs, 1,
                                 &softmax_params, sizeof(softmax_params), false, &attn_weights),
                err, fail_free_scores);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, attn_weights), err, fail_free_attn_weights);

    // Step 8: Compute output = attn_weights @ V
    // output shape: [batch, num_head, seq_len, head_dim]
    uint32_t output_shapes[4] = {batch, params.num_head, seq_len, head_dim};
    struct vkllm_tensor *output_srcs[] = {attn_weights, V_ref};
    matmul_params.scale = 1.0f;

    snprintf(scope_buf, sizeof(scope_buf), "%s.attn_output", name);
    _CHECK_JUMP(vkllm_tensor_new(context, "attn_output", output_shapes, input->dtype, VKLLM_OP_MATMUL, output_srcs, 2,
                                 &matmul_params, sizeof(matmul_params), false, &output),
                err, fail_free_attn_weights);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, output), err, fail_free_attn_output);

    // Step 9: Reshape output back to [batch, 1, seq_len, num_head*head_dim]
    // First permute output from [batch, num_head, seq_len, head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t output_permute_axis[4] = {
        0, 2, 1, 3}; // (batch, num_head, seq_len, head_dim) -> (batch, seq_len, num_head, head_dim)
    struct vkllm_tensor *output_ref = NULL;
    _CHECK_JUMP(vkllm_tensor_copy_ref(context, output, &output_ref), err, fail_free_attn_output);
    _CHECK_JUMP(vkllm_tensor_permute(context, output_ref, output_permute_axis), err, fail_free_output_ref);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, output_ref), err, fail_free_output_ref);

    struct vkllm_tensor *concated_heads = NULL;
    struct vkllm_tensor *concated_heads_ref = NULL;
    snprintf(scope_buf, sizeof(scope_buf), "%s.concated_heads", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, output_ref->shapes, output_ref->dtype, VKLLM_OP_COPY, &output_ref,
                                 1, NULL, 0, false, &concated_heads),
                err, fail_free_cocnated_heads);
    _CHECK_JUMP(vkllm_tensor_copy_ref(context, concated_heads, &concated_heads_ref), err, fail_free_cocnated_heads);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, concated_heads), err, fail_free_cocnated_heads);

    // Then reshape to [batch, 1, seq_len, num_head*head_dim]
    uint32_t concated_heads_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    _CHECK_JUMP(vkllm_tensor_reshape(context, concated_heads_ref, concated_heads_shapes), err,
                fail_free_concated_heads_ref);

    struct vkllm_tensor *final_output = NULL;
    uint32_t final_output_shapes[] = {batch, 1, seq_len, hidden_dim};
    struct vkllm_tensor *final_output_srcs[] = {concated_heads_ref, params.WO};
    struct vkllm_op_matmul_params final_output_params = {.scale = 1.0, .act = 0};
    snprintf(scope_buf, sizeof(scope_buf), "%s.final_output", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, final_output_shapes, concated_heads->dtype, VKLLM_OP_MATMUL,
                                 final_output_srcs, 2, &final_output_params, sizeof(final_output_params), false,
                                 &final_output),
                err, fail_free_cocnated_heads);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, final_output), err, fail_free_final_output);

    struct vkllm_tensor *final_output_add;
    struct vkllm_tensor *output_add_srcs[] = {input, final_output};
    int32_t bin_op = 0;
    snprintf(scope_buf, sizeof(scope_buf), "%s.final_output_add", name);
    _CHECK_JUMP(vkllm_tensor_new(context, scope_buf, final_output->shapes, final_output->dtype, VKLLM_OP_BIN,
                                 output_add_srcs, 2, &bin_op, sizeof(bin_op), false, &final_output_add),
                err, fail_free_final_output);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, final_output_add), err, fail_free_output_add);

    return VKLLM_ERR_OK;

fail_free_output_add:
    vkllm_tensor_free(context, final_output_add);
fail_free_final_output:
    vkllm_tensor_free(context, final_output);
fail_free_concated_heads_ref:
    vkllm_tensor_free(context, concated_heads_ref);
fail_free_cocnated_heads:
    vkllm_tensor_free(context, concated_heads);
fail_free_output_ref:
    vkllm_tensor_free(context, output_ref);
fail_free_attn_output:
    vkllm_tensor_free(context, output);
fail_free_attn_weights:
    vkllm_tensor_free(context, attn_weights);
fail_free_scores:
    vkllm_tensor_free(context, scores);
fail_free_RK:
    vkllm_tensor_free(context, RK);
fail_free_RQ:
    vkllm_tensor_free(context, RQ);
fail_free_V_ref:
    vkllm_tensor_free(context, V_ref);
fail_free_K_ref:
    vkllm_tensor_free(context, K_ref);
fail_free_Q_ref:
    vkllm_tensor_free(context, Q_ref);
fail_free_V:
    vkllm_tensor_free(context, V);
fail_free_K:
    vkllm_tensor_free(context, K);
fail_free_Q:
    vkllm_tensor_free(context, Q);
fail_free_norm:
    vkllm_tensor_free(context, norm);
    return err;
}

vkllm_err_t vkllm_llama2_build_transformer_block(struct vkllm_context *context, struct vkllm_graph *graph,
                                                 struct vkllm_tensor *input,
                                                 struct vkllm_llama2_transformer_block_params params, const char *name)
{
    _CHECK_ARGS(context && graph && input);

    char scope_buf[256];
    snprintf(scope_buf, sizeof(scope_buf), "%s.attn", name);
    _CHECK(vkllm_llama2_build_self_attn_layer(context, graph, input, params.attn, scope_buf));
    snprintf(scope_buf, sizeof(scope_buf), "%s.ffn", name);
    _CHECK(vkllm_llama2_build_ffn_layer(context, graph, input, params.ffn, scope_buf));

    return VKLLM_ERR_OK;
}
