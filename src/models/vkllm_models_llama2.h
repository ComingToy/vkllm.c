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

struct vkllm_token
{
    char *text;
    float score;
    int32_t type;
};

VKLLM_DEF_ARRAY(token, struct vkllm_token);
struct vkllm_array_token_id
{
    size_t alloc_n;
    size_t used_n;
    uint32_t *data;
};
static inline vkllm_err_t vkllm_array_token_id_new(struct vkllm_array_token_id **arr, size_t init)
{
    size_t alloc_bytes = sizeof(**arr);
    *arr = (struct vkllm_array_token_id *)malloc(alloc_bytes);
    if (*arr == ((void *)0))
    {
        return VKLLM_ERR_ALLOC;
    }
    (*arr)->data = ((void *)0);
    (*arr)->alloc_n = init;
    (*arr)->used_n = 0;
    if (init > 0)
    {
        do
        {
            ((*arr)->data) = (uint32_t *)malloc(sizeof(uint32_t) * (init));
            if (!((*arr)->data))
                return VKLLM_ERR_ALLOC;
        } while (0);
    }
    return VKLLM_ERR_OK;
}
static inline vkllm_err_t vkllm_array_token_id_copy(struct vkllm_array_token_id *src, struct vkllm_array_token_id **dst)
{
    *dst = (struct vkllm_array_token_id *)malloc(sizeof(*src));
    if (*dst == ((void *)0))
    {
        return VKLLM_ERR_ALLOC;
    }
    do
    {
        ((*dst)->data) = (uint32_t *)malloc(sizeof(uint32_t) * (src->alloc_n));
        if (!((*dst)->data))
            return VKLLM_ERR_ALLOC;
    } while (0);
    (*dst)->alloc_n = src->alloc_n;
    (*dst)->used_n = src->used_n;
    memcpy((*dst)->data, src->data, sizeof(uint32_t) * src->used_n);
    return VKLLM_ERR_OK;
}
static inline vkllm_err_t vkllm_array_token_id_append(struct vkllm_array_token_id *arr, uint32_t element)
{
    if (arr->used_n >= arr->alloc_n)
    {
        uint32_t *data = ((void *)0);
        do
        {
            (data) = (uint32_t *)malloc(sizeof(uint32_t) * (arr->alloc_n * 2));
            if (!(data))
                return VKLLM_ERR_ALLOC;
        } while (0);
        memcpy(data, arr->data, arr->alloc_n * sizeof(uint32_t));
        free(arr->data);
        arr->data = data;
        arr->alloc_n *= 2;
    }
    arr->data[arr->used_n++] = element;
    return VKLLM_ERR_OK;
}
static inline void vkllm_array_token_id_free(struct vkllm_array_token_id *arr)
{
    if (!arr)
        return;
    if (arr->data)
        free(arr->data);
    free(arr);
};

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
        uint32_t bos_token_id;
        uint32_t eos_token_id;
        uint32_t padding_token_id;
        bool add_bos_token;
        bool add_eos_token;
        struct vkllm_array_token *tokens;
    } meta;

    struct
    {
        struct vkllm_tensor *tok_embed_weights;
        struct vkllm_array_block_weights *blocks;
        struct vkllm_tensor *output_norm_weight;
        struct vkllm_tensor *output_weight;
    } weights;
};

extern vkllm_err_t vkllm_models_llama2_load(struct vkllm_context *context, struct vkllm_models_llama2 *model,
                                            const char *file);
extern vkllm_err_t vkllm_models_llama2_free(struct vkllm_context *context, struct vkllm_models_llama2 *model);

extern vkllm_err_t vkllm_models_llama2_build_graph(struct vkllm_context *context, struct vkllm_models_llama2 *model,
                                                   struct vkllm_tensor *input_toks, struct vkllm_graph *graph);
extern vkllm_err_t vkllm_models_llama2_tokenize(struct vkllm_models_llama2 *model, const char *sentence,
                                                struct vkllm_array_token_id **token_ids);

#endif
