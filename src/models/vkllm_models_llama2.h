#ifndef __VKLLM_MODELS_LLAMA2_H__
#define __VKLLM_MODELS_LLAMA2_H__

#include "src/core/vkllm_array.h"
#include "vkllm_llama2_layers.h"
struct block_weights
{
    struct vkllm_tensor *attn_norm_weight;
    struct vkllm_tensor *ffn_norm_weight;
    struct vkllm_tensor *WQ;
    struct vkllm_tensor *WK;
    struct vkllm_tensor *WV;
    struct vkllm_tensor *WO;
    struct vkllm_tensor *WU;
    struct vkllm_tensor *WG;
    struct vkllm_tensor *WD;
};

VKLLM_DEF_ARRAY(block_weights, struct block_weights *);

struct vkllm_graph;

struct vkllm_models_llama2
{
    struct
    {
        uint32_t block_count;
        uint32_t context_length;
        uint32_t embedding_length;
        float rope_freq_base;
        float layer_norm_rms_epsilon;
        uint32_t key_length;
        uint32_t value_length;
        uint32_t vocab_size;
        uint32_t head_count;
        uint32_t head_count_kv;
    } meta;

    struct
    {
        struct vkllm_tensor *tok_embed_weights;
        struct vkllm_array_block_weights *blocks;
        struct vkllm_tensor *output_norm_weight;
        struct vkllm_tensor *output_weight;
    } weights;

    struct vkllm_graph *graph;
};

extern vkllm_err_t vkllm_models_llama2_load(struct vkllm_context *context, struct vkllm_models_llama2 *model,
                                            const char *file);
extern vkllm_err_t vkllm_models_llama2_free(struct vkllm_context *context, struct vkllm_models_llama2 *model);

extern vkllm_err_t vkllm_models_llama2_build_model(struct vkllm_context *context, struct vkllm_models_llama2 *model,
                                                   struct vkllm_tensor *input_toks);

#endif
