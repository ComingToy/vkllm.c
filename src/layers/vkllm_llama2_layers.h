#ifndef __VKLLM_LLAMA2_LAYERS_H___
#define __VKLLM_LLAMA2_LAYERS_H___

#include "../core/vkllm_errors.h"
#include "../core/vkllm_op_rope.h"
#include <stdint.h>
struct vkllm_context;
struct vkllm_graph;
struct vkllm_tensor;

struct vkllm_llama2_self_attn_layer_params
{
    struct vkllm_tensor *WK, *WQ, *WV;
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

extern vkllm_err_t vkllm_llama2_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                                      struct vkllm_tensor *input,
                                                      struct vkllm_llama2_self_attn_layer_params params);

extern vkllm_err_t vkllm_llama2_build_ffn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                                struct vkllm_tensor *input,
                                                struct vkllm_llama2_ffn_layer_params params);
#endif
