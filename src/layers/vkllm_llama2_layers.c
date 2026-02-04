#include "vkllm_llama2_layers.h"
#include "../core/vkllm_context.h"
#include "../core/vkllm_graph.h"
#include "../core/vkllm_op_bin.h"
#include "../core/vkllm_op_matmul.h"
#include "../core/vkllm_op_softmax.h"
#include "../core/vkllm_tensor.h"
#include <math.h>

vkllm_err_t vkllm_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                        struct vkllm_tensor *input, struct vkllm_tensor *WQ, struct vkllm_tensor *WK,
                                        struct vkllm_tensor *WV)
{
    _CHECK_ARGS(context != NULL);
    _CHECK_ARGS(graph != NULL);
    _CHECK_ARGS(input != NULL);
    _CHECK_ARGS(WQ != NULL);
    _CHECK_ARGS(WK != NULL);
    _CHECK_ARGS(WV != NULL);

    // Get dimensions from input: [batch, 1, seq_len, hidden_dim]
    // Using 4D tensor format: [B, C, M, K]
    uint32_t batch = input->shapes[0];
    uint32_t seq_len = input->shapes[2];

    // Get head_dim from WQ: [1, num_head, head_dim, hidden_dim]
    uint32_t num_head = WQ->shapes[1];
    uint32_t head_dim = WQ->shapes[2];

    // Step 1: Q = input × WQ
    // input: [batch, 1, seq_len, hidden_dim]
    // WQ: [1, num_head, head_dim, hidden_dim]
    // Q: [batch, num_head, seq_len, head_dim]
    uint32_t Q_shapes[4] = {batch, num_head, seq_len, head_dim};
    struct vkllm_tensor *Q_srcs[] = {input, WQ};
    struct vkllm_tensor *Q = NULL;
    _CHECK(vkllm_tensor_new(context, "Q", Q_shapes, input->dtype, VKLLM_OP_MATMUL, Q_srcs, 2, NULL, 0, false, &Q));
    _CHECK(vkllm_graph_add_node(context, graph, Q));

    // Step 2: K = input × WK
    // K: [batch, num_head, seq_len, head_dim]
    uint32_t K_shapes[4] = {batch, num_head, seq_len, head_dim};
    struct vkllm_tensor *K_srcs[] = {input, WK};
    struct vkllm_tensor *K = NULL;
    _CHECK(vkllm_tensor_new(context, "K", K_shapes, input->dtype, VKLLM_OP_MATMUL, K_srcs, 2, NULL, 0, false, &K));
    _CHECK(vkllm_graph_add_node(context, graph, K));

    // Step 3: V = input × WV
    // V: [batch, num_head, seq_len, head_dim]
    uint32_t V_shapes[4] = {batch, num_head, seq_len, head_dim};
    struct vkllm_tensor *V_srcs[] = {input, WV};
    struct vkllm_tensor *V = NULL;
    _CHECK(vkllm_tensor_new(context, "V", V_shapes, input->dtype, VKLLM_OP_MATMUL, V_srcs, 2, NULL, 0, false, &V));
    _CHECK(vkllm_graph_add_node(context, graph, V));

    // Step 5: scores = Q × K^T
    // Q: [batch, num_head, seq_len, head_dim]
    // scores: [batch, channels, seq_len, seq_len]
    uint32_t scores_shapes[4] = {batch, num_head, seq_len, seq_len};
    struct vkllm_tensor *scores_srcs[] = {Q, K};
    struct vkllm_tensor *scores = NULL;
    float scale_factor = 1.0f / sqrtf((float)head_dim);

    _CHECK(vkllm_tensor_new(context, "scores", scores_shapes, vkllm_dtype_float32, VKLLM_OP_MATMUL, scores_srcs, 2,
                            NULL, 0, false, &scores));
    _CHECK(vkllm_graph_add_node(context, graph, scores));

    // Step 7: attn_weights = softmax(scaled_scores)
    // Apply softmax along the last dimension (seq_len)
    uint32_t attn_weights_shapes[4] = {batch, num_head, seq_len, seq_len};
    struct vkllm_tensor *attn_weights_srcs[] = {scores};
    struct vkllm_tensor *attn_weights = NULL;
    _CHECK(vkllm_tensor_new(context, "attn_weights", attn_weights_shapes, vkllm_dtype_float32, VKLLM_OP_SOFTMAX,
                            attn_weights_srcs, 1, NULL, 0, false, &attn_weights));
    _CHECK(vkllm_graph_add_node(context, graph, attn_weights));

    // Step 8: output = attn_weights × V
    // attn_weights: [batch, channels, seq_len, seq_len]
    // V: [batch, channels, seq_len, head_dim]
    // output: [batch, channels, seq_len, head_dim]
    uint32_t output_shapes[4] = {batch, num_head, seq_len, head_dim};
    struct vkllm_tensor *output_srcs[] = {attn_weights, V};
    struct vkllm_tensor *output = NULL;
    _CHECK(vkllm_tensor_new(context, "attn_output", output_shapes, vkllm_dtype_float32, VKLLM_OP_MATMUL, output_srcs, 2,
                            NULL, 0, false, &output));
    _CHECK(vkllm_graph_add_node(context, graph, output));

    // Set the output of the graph
    _CHECK(vkllm_graph_set_output(context, graph, output));

    return VKLLM_ERR_OK;
}
