#ifndef __VKLLM_LLAMA2_LAYERS_H___
#define __VKLLM_LLAMA2_LAYERS_H___

#include "../core/vkllm_errors.h"
#include "../core/vkllm_op_rope.h"
#include <stdint.h>
struct vkllm_context;
struct vkllm_graph;
struct vkllm_tensor;
struct vkllm_kvcache;

struct vkllm_llama2_self_attn_layer_params
{
    struct vkllm_tensor *WK, *WQ, *WV, *WO;
    struct vkllm_tensor *norm_weight;
    float norm_power;
    float norm_eps;
    float freq_base;
    uint32_t offsets;
    uint32_t num_head;
};

struct vkllm_llama2_ffn_layer_params
{
    struct vkllm_tensor *WU, *WG, *WD;
    struct vkllm_tensor *norm_weight;
    float norm_power;
    float norm_eps;
};

struct vkllm_llama2_transformer_block_params
{
    struct vkllm_llama2_ffn_layer_params ffn;
    struct vkllm_llama2_self_attn_layer_params attn;
};

// Ownership: both builders add every tensor they create to graph->nodes
// (transferring ownership to the graph). On error the caller frees the graph.
// The layer's result tensor is returned via `*output` (owned by the graph).
extern vkllm_err_t vkllm_llama2_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                                      struct vkllm_kvcache *kvcache, struct vkllm_tensor *input,
                                                      struct vkllm_llama2_self_attn_layer_params params,
                                                      uint32_t block_idx, struct vkllm_tensor **output);

extern vkllm_err_t vkllm_llama2_build_ffn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                                struct vkllm_tensor *input, struct vkllm_llama2_ffn_layer_params params,
                                                uint32_t block_idx, struct vkllm_tensor **output);

extern vkllm_err_t vkllm_llama2_build_transformer_block(struct vkllm_context *context, struct vkllm_graph *graph,
                                                        struct vkllm_kvcache *kvcache, struct vkllm_tensor *input,
                                                        struct vkllm_llama2_transformer_block_params params,
                                                        uint32_t block_idx, struct vkllm_tensor **output);
#endif
