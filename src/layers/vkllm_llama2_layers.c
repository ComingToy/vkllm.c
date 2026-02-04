#include "vkllm_llama2_layers.h"
#include "../core/vkllm_context.h"
#include "../core/vkllm_graph.h"
#include "../core/vkllm_op_softmax.h"
#include "../core/vkllm_tensor.h"
#include "src/core/vkllm_common.h"
#include <math.h>

vkllm_err_t vkllm_llama2_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                               struct vkllm_tensor *input, struct vkllm_tensor *WQ,
                                               struct vkllm_tensor *WK, struct vkllm_tensor *WV,
                                               struct vkllm_op_rope_params rope_params, const uint32_t num_head)
{
    _CHECK_ARGS(context && graph && input && WQ && WK && WV);

    uint32_t hidden_dim = WQ->shapes[3];              // hidden_dim from weight matrix
    uint32_t num_head_times_head_dim = WQ->shapes[2]; // num_head * head_dim
    uint32_t head_dim = num_head_times_head_dim / num_head;

    // Validate weight shapes
    _CHECK_ARGS(WK->shapes[0] == 1 && WK->shapes[1] == 1);
    _CHECK_ARGS(WV->shapes[0] == 1 && WV->shapes[1] == 1);
    _CHECK_ARGS(WK->shapes[2] == num_head_times_head_dim && WK->shapes[3] == hidden_dim);
    _CHECK_ARGS(WV->shapes[2] == num_head_times_head_dim && WV->shapes[3] == hidden_dim);

    // Get input dimensions
    uint32_t batch = input->shapes[0];
    uint32_t seq_len = input->shapes[2];

    // Validate input shape
    _CHECK_ARGS(input->shapes[3] == hidden_dim);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_tensor *Q = NULL, *K = NULL, *V = NULL;
    struct vkllm_tensor *RQ = NULL, *RK = NULL;
    struct vkllm_tensor *scores = NULL, *attn_weights = NULL, *output = NULL;

    // Step 1: Compute Q = input @ WQ^T
    // Q shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t Q_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *Q_srcs[] = {input, WQ};
    float scale_q = 1.0f;
    err = vkllm_tensor_new(context, "Q", Q_shapes, input->dtype, VKLLM_OP_MATMUL, Q_srcs, 2, &scale_q, sizeof(scale_q),
                           false, &Q);
    _CHECK(err);
    _CHECK(vkllm_graph_add_node(context, graph, Q));

    // Step 2: Compute K = input @ WK^T
    // K shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t K_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *K_srcs[] = {input, WK};
    float scale_k = 1.0f;
    err = vkllm_tensor_new(context, "K", K_shapes, input->dtype, VKLLM_OP_MATMUL, K_srcs, 2, &scale_k, sizeof(scale_k),
                           false, &K);
    _CHECK(err);
    _CHECK(vkllm_graph_add_node(context, graph, K));

    // Step 3: Compute V = input @ WV^T
    // V shape: [batch, 1, seq_len, num_head*head_dim]
    uint32_t V_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    struct vkllm_tensor *V_srcs[] = {input, WV};
    float scale_v = 1.0f;
    err = vkllm_tensor_new(context, "V", V_shapes, input->dtype, VKLLM_OP_MATMUL, V_srcs, 2, &scale_v, sizeof(scale_v),
                           false, &V);
    _CHECK(err);
    _CHECK(vkllm_graph_add_node(context, graph, V));

    // Step 4: Reshape Q, K, V to split heads
    // First reshape Q from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t Q_reshaped_shapes[4] = {batch, seq_len, num_head, head_dim};
    err = vkllm_tensor_reshape(context, Q, Q_reshaped_shapes);
    _CHECK(err);

    // Then permute Q to [batch, num_head, seq_len, head_dim]
    uint32_t Q_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    err = vkllm_tensor_permute(context, Q, Q_permute_axis);
    _CHECK(err);

    // First reshape K from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t K_reshaped_shapes[4] = {batch, seq_len, num_head, head_dim};
    err = vkllm_tensor_reshape(context, K, K_reshaped_shapes);
    _CHECK(err);

    // Then permute K to [batch, num_head, seq_len, head_dim]
    uint32_t K_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    err = vkllm_tensor_permute(context, K, K_permute_axis);
    _CHECK(err);

    // First reshape V from [batch, 1, seq_len, num_head*head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t V_reshaped_shapes[4] = {batch, seq_len, num_head, head_dim};
    err = vkllm_tensor_reshape(context, V, V_reshaped_shapes);
    _CHECK(err);

    // Then permute V to [batch, num_head, seq_len, head_dim]
    uint32_t V_permute_axis[4] = {0, 2, 1,
                                  3}; // (batch, seq_len, num_head, head_dim) -> (batch, num_head, seq_len, head_dim)
    err = vkllm_tensor_permute(context, V, V_permute_axis);
    _CHECK(err);

    _CHECK(vkllm_tensor_new(context, "RQ", Q->shapes, Q->dtype, VKLLM_OP_ROPE, &Q, 1, &rope_params, sizeof(rope_params),
                            false, &RQ));

    _CHECK(vkllm_tensor_new(context, "RK", K->shapes, K->dtype, VKLLM_OP_ROPE, &K, 1, &rope_params, sizeof(rope_params),
                            false, &RK));

    // Step 6: Compute scores = Q @ K^T / sqrt(head_dim)
    // scores shape: [batch, num_head, seq_len, seq_len]
    uint32_t scores_shapes[4] = {batch, num_head, seq_len, seq_len};
    struct vkllm_tensor *scores_srcs[] = {RQ, RK};
    float scale_scores = 1.0f / sqrtf((float)head_dim);
    err = vkllm_tensor_new(context, "scores", scores_shapes, input->dtype, VKLLM_OP_MATMUL, scores_srcs, 2,
                           &scale_scores, sizeof(scale_scores), false, &scores);
    _CHECK(err);
    _CHECK(vkllm_graph_add_node(context, graph, scores));

    // Step 7: Apply softmax to get attention weights
    // attn_weights shape: [batch, num_head, seq_len, seq_len]
    struct vkllm_tensor *softmax_srcs[] = {scores};
    struct vkllm_op_softmax_params softmax_params = {.seq_mask = 1, // No masking
                                                     .offsets = rope_params.offset};
    err = vkllm_tensor_new(context, "attn_weights", scores_shapes, input->dtype, VKLLM_OP_SOFTMAX, softmax_srcs, 1,
                           &softmax_params, sizeof(softmax_params), false, &attn_weights);
    _CHECK(err);
    _CHECK(vkllm_graph_add_node(context, graph, attn_weights));

    // Step 8: Compute output = attn_weights @ V
    // output shape: [batch, num_head, seq_len, head_dim]
    uint32_t output_shapes[4] = {batch, num_head, seq_len, head_dim};
    struct vkllm_tensor *output_srcs[] = {attn_weights, V};
    float scale_output = 1.0f;
    err = vkllm_tensor_new(context, "attn_output", output_shapes, input->dtype, VKLLM_OP_MATMUL, output_srcs, 2,
                           &scale_output, sizeof(scale_output), false, &output);
    _CHECK(err);
    _CHECK(vkllm_graph_add_node(context, graph, output));

    // Step 9: Reshape output back to [batch, 1, seq_len, num_head*head_dim]
    // First permute output from [batch, num_head, seq_len, head_dim] to [batch, seq_len, num_head, head_dim]
    uint32_t output_permute_axis[4] = {
        0, 2, 1, 3}; // (batch, num_head, seq_len, head_dim) -> (batch, seq_len, num_head, head_dim)
    err = vkllm_tensor_permute(context, output, output_permute_axis);
    _CHECK(err);

    // Then reshape to [batch, 1, seq_len, num_head*head_dim]
    uint32_t output_final_shapes[4] = {batch, 1, seq_len, num_head_times_head_dim};
    err = vkllm_tensor_reshape(context, output, output_final_shapes);
    _CHECK(err);

    return VKLLM_ERR_OK;
}
