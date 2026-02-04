#ifndef __VKLLM_LLAMA2_LAYERS_H___
#define __VKLLM_LLAMA2_LAYERS_H___

#include "../core/vkllm_errors.h"
#include "../core/vkllm_op_rope.h"
#include <stdint.h>
struct vkllm_context;
struct vkllm_graph;
struct vkllm_tensor;

extern vkllm_err_t vkllm_llama2_build_self_attn_layer(struct vkllm_context *context, struct vkllm_graph *graph,
                                                      struct vkllm_tensor *input, struct vkllm_tensor *WQ,
                                                      struct vkllm_tensor *WK, struct vkllm_tensor *WV,
                                                      struct vkllm_op_rope_params params, const uint32_t num_head);

#endif
