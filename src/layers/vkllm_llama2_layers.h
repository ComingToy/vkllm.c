#ifndef __VKLLM_LLAMA2_LAYERS_H___
#define __VKLLM_LLAMA2_LAYERS_H___

#include "../core/vkllm_errors.h"
#include <stdint.h>
struct vkllm_context;
struct vkllm_graph;
struct vkllm_tensor;

extern vkllm_err_t vkllm_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                               struct vkllm_tensor *input, struct vkllm_tensor *WQ,
                                               struct vkllm_tensor *WK, struct vkllm_tensor *WV);

#endif
