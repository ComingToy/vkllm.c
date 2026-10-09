#include "vkllm_llama2_layers.h"
#include "../core/vkllm_context.h"
#include "../core/vkllm_graph.h"
#include "../core/vkllm_op_ffn_up_and_gate.h"
#include "../core/vkllm_op_matmul.h"
#include "../core/vkllm_op_rmsnorm.h"
#include "../core/vkllm_op_softmax.h"
#include "../core/vkllm_tensor.h"
#include "src/core/vkllm_common.h"
#include "vkllm_kvcache.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Ownership: every tensor created here is added to `graph->nodes` (which
// takes ownership) before anything can fail afterwards, so there is no
// per-tensor cleanup ladder — on error the caller frees the graph.
// `*output` receives the layer's result tensor (owned by the graph).
vkllm_err_t vkllm_llama2_build_ffn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                         struct vkllm_tensor *input, struct vkllm_llama2_ffn_layer_params params,
                                         uint32_t block_idx, struct vkllm_tensor **output)
{
    _CHECK_ARGS(context && graph && input && params.WD && params.WG && params.WU && output);
    struct vkllm_tensor *norm = NULL, *up_gate = NULL, *down = NULL, *out = NULL;
    struct vkllm_tensor *up = NULL, *gate = NULL, *gate_mul = NULL;
    char scope_buf[128];
    char name[64];

    uint32_t batch = input->shapes[0];
    uint32_t channel = input->shapes[1];
    uint32_t seq_len = input->shapes[2];
    uint32_t up_dim = params.WU->shapes[2];

    snprintf(name, sizeof(name), "block.%u.ffn", block_idx);
    struct vkllm_tensor *norm_srcs[] = {input, params.norm_weight};
    struct vkllm_op_rmsnorm_params norm_params = {.power = params.norm_power, .eps = params.norm_eps};
    snprintf(scope_buf, sizeof(scope_buf), "%s.norm", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, input->shapes, input->dtype, VKLLM_OP_RMSNORM, norm_srcs, 2,
                            &norm_params, sizeof(norm_params), false, &norm));
    _CHECK(vkllm_graph_add_node(context, graph, norm));

    uint32_t ffn_up_shapes[] = {batch, channel, seq_len, up_dim};
    struct vkllm_tensor *down_input = NULL;

    // The fused shader is currently DISABLED by default: on Apple GPUs its
    // 128x128 tile with dual 8x8 accumulators (128 float registers per
    // thread) spills and runs ~45% SLOWER than the separate path at every
    // prompt length measured (95-1490 tokens). Opt in with
    // VKLLM_FUSED_FFN=1 (the op itself is correct — see
    // vkllm_test_op_ffn_up_and_gate and the standalone CPU-reference check).
    const char *fuse = getenv("VKLLM_FUSED_FFN");
    if (seq_len > 1 && fuse && fuse[0] == '1')
    {
        // Prefill: fused up+gate projection — one shader computes
        // up * silu(gate) for the whole tile (saves a dispatch and an
        // intermediate buffer vs. separate matmuls + mul).
        struct vkllm_tensor *up_gate_srcs[] = {norm, params.WU, params.WG};

        snprintf(scope_buf, sizeof(scope_buf), "%s.up_gate", name);
        _CHECK(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, norm->dtype, VKLLM_OP_FFN_UP_AND_GATE,
                                up_gate_srcs, 3, NULL, 0, false, &up_gate));
        _CHECK(vkllm_graph_add_node(context, graph, up_gate));
        down_input = up_gate;
    }
    else
    {
        // Decode (seq_len == 1): the fused shader tiles 128x128 blocks and
        // would waste 127/128 of its work on a single row; use the
        // matrix-vector path instead (up + silu(gate) mat_muls + mul).
        struct vkllm_op_matmul_params up_matmul_params = {.scale = 1.0, .act = 0, .transposed_b = true};
        snprintf(scope_buf, sizeof(scope_buf), "%s.up", name);
        _CHECK(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, norm->dtype, VKLLM_OP_MATMUL,
                                (struct vkllm_tensor *[]){norm, params.WU}, 2, &up_matmul_params,
                                sizeof(up_matmul_params), false, &up));
        _CHECK(vkllm_graph_add_node(context, graph, up));

        struct vkllm_op_matmul_params gate_matmul_params = {.scale = 1.0, .act = 1, .transposed_b = true};
        snprintf(scope_buf, sizeof(scope_buf), "%s.gate", name);
        _CHECK(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, norm->dtype, VKLLM_OP_MATMUL,
                                (struct vkllm_tensor *[]){norm, params.WG}, 2, &gate_matmul_params,
                                sizeof(gate_matmul_params), false, &gate));
        _CHECK(vkllm_graph_add_node(context, graph, gate));

        int32_t bin_op = 2; // times
        snprintf(scope_buf, sizeof(scope_buf), "%s.mul", name);
        _CHECK(vkllm_tensor_new(context, scope_buf, ffn_up_shapes, gate->dtype, VKLLM_OP_BIN,
                                (struct vkllm_tensor *[]){up, gate}, 2, &bin_op, sizeof(bin_op), false, &gate_mul));
        _CHECK(vkllm_graph_add_node(context, graph, gate_mul));
        down_input = gate_mul;
    }

    struct vkllm_tensor *down_srcs[] = {down_input, params.WD, input};
    // act == 2 fuses the residual add into the down-projection GEMV (decode
    // path only; prefill keeps the separate add so batched matmul applies).
    struct vkllm_op_matmul_params down_params = {.scale = 1.0, .act = seq_len == 1 ? 2 : 0, .transposed_b = true};
    uint32_t ffn_down_shapes[] = {batch, channel, seq_len, params.WD->shapes[2]};
    snprintf(scope_buf, sizeof(scope_buf), "%s.down", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, ffn_down_shapes, down_input->dtype, VKLLM_OP_MATMUL, down_srcs,
                            seq_len == 1 ? 3 : 2, &down_params, sizeof(down_params), false, &down));
    _CHECK(vkllm_graph_add_node(context, graph, down));

    if (seq_len == 1)
    {
        // The fused GEMV already added the residual.
        *output = down;
        return VKLLM_ERR_OK;
    }

    struct vkllm_tensor *output_srcs[] = {input, down};
    int32_t add_op = 0;

    snprintf(scope_buf, sizeof(scope_buf), "%s.add", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, input->shapes, input->dtype, VKLLM_OP_BIN, output_srcs, 2, &add_op,
                            sizeof(add_op), false, &out));
    _CHECK(vkllm_graph_add_node(context, graph, out));

    *output = out;
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_llama2_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                               struct vkllm_kvcache *kvcache, struct vkllm_tensor *input,
                                               struct vkllm_llama2_self_attn_layer_params params, uint32_t block_idx,
                                               struct vkllm_tensor **output)
{
    _CHECK_ARGS(context && graph && input && params.WQ && params.WK && params.WV);

    struct vkllm_op_rope_params rope_params = {.base = params.freq_base, .offset = params.offsets, .neox_style = false};

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

    struct vkllm_tensor *norm = NULL;
    struct vkllm_tensor *Q = NULL, *K = NULL, *V = NULL;
    struct vkllm_tensor *RQ = NULL, *RK = NULL;
    struct vkllm_tensor *scores = NULL, *attn_weights = NULL, *out = NULL;
    char scope_buf[128];
    char name[64];

    snprintf(name, sizeof(name), "block.%u.attn", block_idx);

    struct vkllm_tensor *norm_srcs[] = {input, params.norm_weight};
    struct vkllm_op_rmsnorm_params norm_params = {.power = params.norm_power, .eps = params.norm_eps};
    snprintf(scope_buf, sizeof(scope_buf), "%s.norm", name);

    _CHECK(vkllm_tensor_new(context, scope_buf, input->shapes, input->dtype, VKLLM_OP_RMSNORM, norm_srcs, 2,
                            &norm_params, sizeof(norm_params), false, &norm));
    _CHECK(vkllm_graph_add_node(context, graph, norm));

    // Step 1: Compute Q = input @ WQ^T
    // Q shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t Q_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *Q_srcs[] = {norm, params.WQ};
    struct vkllm_op_matmul_params matmul_params = {.act = 0, .scale = 1.0f, .transposed_b = true};
    snprintf(scope_buf, sizeof(scope_buf), "%s.Q", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, Q_shapes, norm->dtype, VKLLM_OP_MATMUL, Q_srcs, 2, &matmul_params,
                            sizeof(matmul_params), false, &Q));
    _CHECK(vkllm_graph_add_node(context, graph, Q));

    // Step 2: Compute K = input @ WK^T
    // K shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t K_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *K_srcs[] = {norm, params.WK};
    snprintf(scope_buf, sizeof(scope_buf), "%s.K", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, K_shapes, norm->dtype, VKLLM_OP_MATMUL, K_srcs, 2, &matmul_params,
                            sizeof(matmul_params), false, &K));
    _CHECK(vkllm_graph_add_node(context, graph, K));

    // Step 3: Compute V = input @ WV^T
    // V shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t V_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *V_srcs[] = {norm, params.WV};
    snprintf(scope_buf, sizeof(scope_buf), "%s.V", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, V_shapes, norm->dtype, VKLLM_OP_MATMUL, V_srcs, 2, &matmul_params,
                            sizeof(matmul_params), false, &V));
    _CHECK(vkllm_graph_add_node(context, graph, V));

    // Step 4: Reshape Q, K, V to split heads
    // First reshape Q from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t Q_reshaped_shapes[4] = {batch, seq_len, params.num_head, head_dim};
    struct vkllm_tensor *Q_ref = NULL;
    _CHECK(vkllm_tensor_copy_ref(context, Q, &Q_ref));
    _CHECK(vkllm_graph_add_node(context, graph, Q_ref));
    _CHECK(vkllm_tensor_reshape(context, Q_ref, Q_reshaped_shapes));

    // Then permute Q to [batch, num_head, seq_len, head_dim]
    uint32_t Q_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    _CHECK(vkllm_tensor_permute(context, Q_ref, Q_permute_axis));

    // First reshape K from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t K_reshaped_shapes[4] = {batch, seq_len, params.num_head, head_dim};
    struct vkllm_tensor *K_ref = NULL;
    _CHECK(vkllm_tensor_copy_ref(context, K, &K_ref));
    _CHECK(vkllm_graph_add_node(context, graph, K_ref));
    _CHECK(vkllm_tensor_reshape(context, K_ref, K_reshaped_shapes));

    // Then permute K to [batch, num_head, seq_len, head_dim]
    uint32_t K_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    _CHECK(vkllm_tensor_permute(context, K_ref, K_permute_axis));

    // First reshape V from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t V_reshaped_shapes[4] = {batch, seq_len, params.num_head, head_dim};
    struct vkllm_tensor *V_ref = NULL;
    _CHECK(vkllm_tensor_copy_ref(context, V, &V_ref));
    _CHECK(vkllm_graph_add_node(context, graph, V_ref));
    _CHECK(vkllm_tensor_reshape(context, V_ref, V_reshaped_shapes));

    // Then permute V to [batch, num_head, seq_len, head_dim]
    uint32_t V_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    _CHECK(vkllm_tensor_permute(context, V_ref, V_permute_axis));

    snprintf(scope_buf, sizeof(scope_buf), "%s.RQ", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, Q_ref->shapes, Q_ref->dtype, VKLLM_OP_ROPE, &Q_ref, 1, &rope_params,
                            sizeof(rope_params), false, &RQ));
    _CHECK(vkllm_graph_add_node(context, graph, RQ));

    snprintf(scope_buf, sizeof(scope_buf), "%s.RK", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, K_ref->shapes, K_ref->dtype, VKLLM_OP_ROPE, &K_ref, 1, &rope_params,
                            sizeof(rope_params), false, &RK));
    _CHECK(vkllm_graph_add_node(context, graph, RK));

    // kvcache_bind replaces RK/V_ref with persistent views of the layer's
    // caches. The views are owned by the kvcache (REF tensors, nothing to
    // execute); the cache tensors themselves (UPDATE_ROWS ops) are registered
    // as external graph nodes so they run with the graph but stay owned by
    // the model.
    _CHECK(vkllm_kvcache_bind(context, kvcache, graph, &RK, &V_ref, block_idx, params.offsets));

    // Step 6: Compute scores = Q @ K^T / sqrt(head_dim)
    // scores shape: [batch, num_head, seq_len, key_num]
    uint32_t key_num = RK->shapes[2];
    // Allocate at FULL context length so that every later step (longer
    // sequences) is a shrink — tensor_view only allows shrinking. The view
    // is immediately reduced to the current key_num below.
    uint32_t scores_alloc_shapes[4] = {batch, params.num_head, seq_len, kvcache->kcache_shape[2]};
    uint32_t scores_shapes[4] = {batch, params.num_head, seq_len, key_num};
    struct vkllm_tensor *scores_srcs[] = {RQ, RK};
    // B (RK view) is [N=cur_len, K=head_dim] rows → transposed layout.
    matmul_params.scale = 1.0f / sqrtf((float)head_dim);
    matmul_params.transposed_b = true;
    snprintf(scope_buf, sizeof(scope_buf), "%s.scores", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, scores_alloc_shapes, input->dtype, VKLLM_OP_MATMUL, scores_srcs, 2,
                            &matmul_params, sizeof(matmul_params), false, &scores));
    _CHECK(vkllm_graph_add_node(context, graph, scores));
    _CHECK(vkllm_tensor_view(context, scores, scores_shapes));

    // Step 7: Apply softmax to get attention weights
    // attn_weights shape: [batch, num_head, seq_len, seq_len]
    struct vkllm_tensor *softmax_srcs[] = {scores};
    struct vkllm_op_softmax_params softmax_params = {.seq_mask = 1, // No masking
                                                     .offsets = rope_params.offset};

    snprintf(scope_buf, sizeof(scope_buf), "%s.attn_weights", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, scores_alloc_shapes, input->dtype, VKLLM_OP_SOFTMAX, softmax_srcs, 1,
                            &softmax_params, sizeof(softmax_params), false, &attn_weights));
    _CHECK(vkllm_graph_add_node(context, graph, attn_weights));
    _CHECK(vkllm_tensor_view(context, attn_weights, scores_shapes));

    // Step 8: Compute output = attn_weights @ V
    // output shape: [batch, num_head, seq_len, head_dim]
    uint32_t output_shapes[4] = {batch, params.num_head, seq_len, head_dim};
    struct vkllm_tensor *output_srcs[] = {attn_weights, V_ref};
    // B (V view) is [K=cur_len, N=head_dim] → row-major K-major layout,
    // not transposed. Must be explicit: shape inference flips when
    // cur_len == head_dim.
    matmul_params.scale = 1.0f;
    matmul_params.transposed_b = false;

    snprintf(scope_buf, sizeof(scope_buf), "%s.attn_output", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, output_shapes, input->dtype, VKLLM_OP_MATMUL, output_srcs, 2,
                            &matmul_params, sizeof(matmul_params), false, &out));
    _CHECK(vkllm_graph_add_node(context, graph, out));

    // Step 9: Reshape output back to [batch, 1, seq_len, num_head*head_dim]
    // First permute output from [batch, num_head, seq_len, head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t output_permute_axis[4] = {
        0, 2, 1, 3}; // (batch, num_head, seq_len, head_dim) -> (batch, seq_len, num_head, head_dim)
    struct vkllm_tensor *output_ref = NULL;
    _CHECK(vkllm_tensor_copy_ref(context, out, &output_ref));
    _CHECK(vkllm_graph_add_node(context, graph, output_ref));
    _CHECK(vkllm_tensor_permute(context, output_ref, output_permute_axis));

    struct vkllm_tensor *concated_heads = NULL;
    struct vkllm_tensor *concated_heads_ref = NULL;
    snprintf(scope_buf, sizeof(scope_buf), "%s.concated_heads", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, output_ref->shapes, output_ref->dtype, VKLLM_OP_COPY, &output_ref, 1,
                            NULL, 0, false, &concated_heads));
    _CHECK(vkllm_graph_add_node(context, graph, concated_heads));
    _CHECK(vkllm_tensor_copy_ref(context, concated_heads, &concated_heads_ref));
    _CHECK(vkllm_graph_add_node(context, graph, concated_heads_ref));

    // Then reshape to [batch, 1, seq_len, num_head*head_dim]
    uint32_t concated_heads_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    _CHECK(vkllm_tensor_reshape(context, concated_heads_ref, concated_heads_shapes));

    struct vkllm_tensor *final_output = NULL;
    uint32_t final_output_shapes[] = {batch, 1, seq_len, hidden_dim};
    struct vkllm_tensor *final_output_srcs[] = {concated_heads_ref, params.WO, input};
    // act == 2 fuses the residual add into the o_proj GEMV epilogue (one
    // dispatch instead of matmul-then-add; decode path only — prefill keeps
    // the separate ops so batched matmul stays available).
    struct vkllm_op_matmul_params final_output_params = {.scale = 1.0, .act = seq_len == 1 ? 2 : 0,
                                                         .transposed_b = true};
    snprintf(scope_buf, sizeof(scope_buf), "%s.final_output", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, final_output_shapes, concated_heads->dtype, VKLLM_OP_MATMUL,
                            final_output_srcs, seq_len == 1 ? 3 : 2, &final_output_params,
                            sizeof(final_output_params), false, &final_output));
    _CHECK(vkllm_graph_add_node(context, graph, final_output));

    struct vkllm_tensor *final_output_add;
    if (seq_len == 1)
    {
        // The fused GEMV already added the residual; the layer output IS the
        // projection result.
        *output = final_output;
        return VKLLM_ERR_OK;
    }

    struct vkllm_tensor *output_add_srcs[] = {input, final_output};
    int32_t bin_op = 0;
    snprintf(scope_buf, sizeof(scope_buf), "%s.final_output_add", name);
    _CHECK(vkllm_tensor_new(context, scope_buf, final_output->shapes, final_output->dtype, VKLLM_OP_BIN,
                            output_add_srcs, 2, &bin_op, sizeof(bin_op), false, &final_output_add));
    _CHECK(vkllm_graph_add_node(context, graph, final_output_add));

    *output = final_output_add;
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_llama2_build_transformer_block(struct vkllm_context *context, struct vkllm_graph *graph,
                                                 struct vkllm_kvcache *kvcache, struct vkllm_tensor *input,
                                                 struct vkllm_llama2_transformer_block_params params,
                                                 uint32_t block_idx, struct vkllm_tensor **output)
{
    _CHECK_ARGS(context && graph && input && output);

    struct vkllm_tensor *attn_output = NULL;
    _CHECK(vkllm_llama2_build_self_attn_layer(context, graph, kvcache, input, params.attn, block_idx, &attn_output));
    _CHECK(vkllm_llama2_build_ffn_layer(context, graph, attn_output, params.ffn, block_idx, output));

    return VKLLM_ERR_OK;
}
