#include "vkllm_models_llama2.h"
#include "../core/vkllm_commands.h"
#include "../core/vkllm_graph.h"
#include "../core/vkllm_op_embedding.h"
#include "../core/vkllm_op_matmul.h"
#include "../core/vkllm_op_rmsnorm.h"
#include "../core/vkllm_tensor.h"
#include "gguflib.h"
#include "vkllm_llama2_layers.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tokenizer_parse_ctx
{
    struct vkllm_array_token *tokens;
    char **token_texts;
    float *scores;
    int32_t *token_types;
    uint64_t count;
};

static void tokenizer_tokens_callback(void *privdata, uint32_t type, union gguf_value *val, uint64_t in_array,
                                      uint64_t array_len)
{
    struct tokenizer_parse_ctx *ctx = (struct tokenizer_parse_ctx *)privdata;
    if (type == GGUF_VALUE_TYPE_STRING && in_array > 0)
    {
        uint64_t idx = in_array - 1;
        if (idx < ctx->count)
        {
            ctx->token_texts[idx] = (char *)malloc(val->string.len + 1);
            memcpy(ctx->token_texts[idx], val->string.string, val->string.len);
            ctx->token_texts[idx][val->string.len] = '\0';
        }
    }
}

static void tokenizer_scores_callback(void *privdata, uint32_t type, union gguf_value *val, uint64_t in_array,
                                      uint64_t array_len)
{
    struct tokenizer_parse_ctx *ctx = (struct tokenizer_parse_ctx *)privdata;
    if (type == GGUF_VALUE_TYPE_FLOAT32 && in_array > 0)
    {
        uint64_t idx = in_array - 1;
        if (idx < ctx->count)
            ctx->scores[idx] = val->float32;
    }
}

static void tokenizer_token_type_callback(void *privdata, uint32_t type, union gguf_value *val, uint64_t in_array,
                                          uint64_t array_len)
{
    struct tokenizer_parse_ctx *ctx = (struct tokenizer_parse_ctx *)privdata;
    if (type == GGUF_VALUE_TYPE_INT32 && in_array > 0)
    {
        uint64_t idx = in_array - 1;
        if (idx < ctx->count)
            ctx->token_types[idx] = val->int32;
    }
}

static vkllm_err_t parse_meta_kv(gguf_ctx *ctx, gguf_key *key, struct vkllm_models_llama2 *model,
                                 struct tokenizer_parse_ctx *tok_ctx)
{
    const char *name = key->name;
    size_t namelen = key->namelen;
    union gguf_value *val = key->val;
    uint32_t type = key->type;

    void (*callback)(void *, uint32_t, union gguf_value *, uint64_t, uint64_t);
    callback = NULL;

    if (strncmp(name, "llama.block_count", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.block_count = val->uint32;
    }
    else if (strncmp(name, "llama.context_length", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.context_length = val->uint32;
    }
    else if (strncmp(name, "llama.embedding_length", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.embedding_length = val->uint32;
    }
    else if (strncmp(name, "llama.rope.freq_base", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_FLOAT32)
            model->meta.rope_freq_base = val->float32;
    }
    else if (strncmp(name, "llama.attention.layer_norm_rms_epsilon", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_FLOAT32)
            model->meta.layer_norm_rms_epsilon = val->float32;
    }
    else if (strncmp(name, "llama.attention.key_length", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.key_length = val->uint32;
    }
    else if (strncmp(name, "llama.attention.value_length", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.value_length = val->uint32;
    }
    else if (strncmp(name, "llama.vocab_size", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.vocab_size = val->uint32;
    }
    else if (strncmp(name, "llama.attention.head_count", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
        {
            model->meta.head_count = val->uint32;
        }
    }
    else if (strncmp(name, "llama.attention.head_count_kv", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
        {
            model->meta.head_count_kv = val->uint32;
        }
    }
    else if (strncmp(name, "tokenizer.ggml.bos_token_id", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.bos_token_id = val->uint32;
    }
    else if (strncmp(name, "tokenizer.ggml.eos_token_id", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.eos_token_id = val->uint32;
    }
    else if (strncmp(name, "tokenizer.ggml.padding_token_id", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_UINT32)
            model->meta.padding_token_id = val->uint32;
    }
    else if (strncmp(name, "tokenizer.ggml.add_bos_token", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_BOOL)
            model->meta.add_bos_token = val->boolval;
    }
    else if (strncmp(name, "tokenizer.ggml.add_eos_token", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_BOOL)
            model->meta.add_eos_token = val->boolval;
    }
    else if (strncmp(name, "tokenizer.ggml.tokens", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_ARRAY && val->array.type == GGUF_VALUE_TYPE_STRING)
        {
            tok_ctx->count = val->array.len;
            tok_ctx->token_texts = (char **)calloc(val->array.len, sizeof(char *));
            callback = tokenizer_scores_callback;
        }
    }
    else if (strncmp(name, "tokenizer.ggml.scores", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_ARRAY && val->array.type == GGUF_VALUE_TYPE_FLOAT32)
        {
            tok_ctx->count = val->array.len;
            tok_ctx->scores = (float *)calloc(val->array.len, sizeof(float));
            callback = tokenizer_scores_callback;
        }
    }
    else if (strncmp(name, "tokenizer.ggml.token_type", namelen) == 0)
    {
        if (type == GGUF_VALUE_TYPE_ARRAY && val->array.type == GGUF_VALUE_TYPE_INT32)
        {
            tok_ctx->count = val->array.len;
            tok_ctx->token_types = (int32_t *)calloc(val->array.len, sizeof(int32_t));
        }
    }

    gguf_do_with_value(ctx, type, val, tok_ctx, 0, 0, callback);
    return VKLLM_ERR_OK;
}

static vkllm_err_t create_tensor_from_gguf(struct vkllm_context *context, struct vkllm_commands *commands,
                                           gguf_tensor *gguf_t, struct vkllm_tensor **ptensor)
{
    vkllm_dtype_t dtype = vkllm_dtype_float16;
    if (gguf_t->type == GGUF_TYPE_F32)
        dtype = vkllm_dtype_float32;
    else if (gguf_t->type == GGUF_TYPE_F16)
        dtype = vkllm_dtype_float16;

    uint32_t shapes[4] = {1, 1, 1, 1};
    for (uint32_t i = 0; i < gguf_t->ndim && i < 4; i++)
    {
        shapes[3 - i] = (uint32_t)gguf_t->dim[i];
    }

    char *name_buf = (char *)malloc(gguf_t->namelen + 1);
    name_buf[gguf_t->namelen] = '\0';
    memcpy(name_buf, gguf_t->name, gguf_t->namelen);

    vkllm_err_t err =
        vkllm_tensor_new(context, name_buf, shapes, dtype, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, ptensor);
    free(name_buf);

    if (err != VKLLM_ERR_OK)
        return err;

    _CHECK(vkllm_commands_begin(context, commands));
    _CHECK(vkllm_commands_upload(context, commands, *ptensor, gguf_t->weights_data, gguf_t->bsize));
    _CHECK(vkllm_commands_end(context, commands));
    _CHECK(vkllm_commands_submit(context, commands));
    _CHECK(vkllm_commands_wait_exec(context, commands));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_models_llama2_load(struct vkllm_context *context, struct vkllm_models_llama2 *model, const char *file)
{
    _CHECK_ARGS(context && model && file);

    memset(model, 0, sizeof(*model));
    vkllm_err_t err = VKLLM_ERR_OK;

    struct tokenizer_parse_ctx tok_ctx = {0};

    gguf_ctx *gguf = gguf_open(file);
    if (!gguf)
    {
        log_error("failed to open gguf file: %s", file);
        return VKLLM_ERR_ARGS;
    }

    struct vkllm_commands *commands = NULL;
    _CHECK_JUMP(vkllm_commands_new(context, &commands), err, cleanup_gguf);

    gguf_key key;
    while (gguf_get_key(gguf, &key))
    {
        parse_meta_kv(gguf, &key, model, &tok_ctx);
    }

    if (tok_ctx.count > 0 && tok_ctx.token_texts)
    {
        _CHECK_JUMP(vkllm_array_token_new(&model->meta.tokens, tok_ctx.count), err, cleanup_commands);
        for (uint64_t i = 0; i < tok_ctx.count; i++)
        {
            struct vkllm_token tok;
            tok.text = tok_ctx.token_texts[i];
            tok.score = tok_ctx.scores ? tok_ctx.scores[i] : 0.0f;
            tok.type = tok_ctx.token_types ? tok_ctx.token_types[i] : 0;
            _CHECK_JUMP(vkllm_array_token_append(model->meta.tokens, tok), err, cleanup_commands);
        }
        free(tok_ctx.token_texts);
        if (tok_ctx.scores)
            free(tok_ctx.scores);
        if (tok_ctx.token_types)
            free(tok_ctx.token_types);
    }

    _CHECK(vkllm_array_block_weights_new(&model->weights.blocks, model->meta.block_count));

    gguf_tensor tensor;
    while (gguf_get_tensor(gguf, &tensor))
    {
        const char *name = tensor.name;
        size_t namelen = tensor.namelen;

        if (strncmp(name, "token_embd.weight", namelen) == 0)
        {
            _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &model->weights.tok_embed_weights), err,
                        cleanup_commands);
        }
        else if (strncmp(name, "output_norm.weight", namelen) == 0)
        {
            _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &model->weights.output_norm_weight), err,
                        cleanup_commands);
        }
        else if (strncmp(name, "output.weight", namelen) == 0)
        {
            _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &model->weights.output_weight), err,
                        cleanup_commands);
        }
        else if (namelen > 4 && strncmp(name, "blk.", 4) == 0)
        {
            char name_buf[128];
            size_t copy_len = namelen < sizeof(name_buf) - 1 ? namelen : sizeof(name_buf) - 1;
            memcpy(name_buf, name, copy_len);
            name_buf[copy_len] = '\0';

            int blk_idx = 0;
            char layer_name[64];
            if (sscanf(name_buf, "blk.%d.%s", &blk_idx, layer_name) == 2)
            {
                while (model->weights.blocks->used_n <= (size_t)blk_idx)
                {
                    struct block_weights *bw = (struct block_weights *)malloc(sizeof(struct block_weights));
                    if (!bw)
                    {
                        err = VKLLM_ERR_ALLOC;
                        goto cleanup_commands;
                    }
                    memset(bw, 0, sizeof(*bw));
                    _CHECK_JUMP(vkllm_array_block_weights_append(model->weights.blocks, bw), err, cleanup_commands);
                }

                struct block_weights *bw = model->weights.blocks->data[blk_idx];
                if (strcmp(layer_name, "attn_norm.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->attn_norm_weight), err,
                                cleanup_commands);
                }
                else if (strcmp(layer_name, "ffn_norm.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->ffn_norm_weight), err,
                                cleanup_commands);
                }
                else if (strcmp(layer_name, "attn_q.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WQ), err, cleanup_commands);
                }
                else if (strcmp(layer_name, "attn_k.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WK), err, cleanup_commands);
                }
                else if (strcmp(layer_name, "attn_v.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WV), err, cleanup_commands);
                }
                else if (strcmp(layer_name, "attn_output.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WO), err, cleanup_commands);
                }
                else if (strcmp(layer_name, "ffn_gate.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WG), err, cleanup_commands);
                }
                else if (strcmp(layer_name, "ffn_up.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WU), err, cleanup_commands);
                }
                else if (strcmp(layer_name, "ffn_down.weight") == 0)
                {
                    _CHECK_JUMP(create_tensor_from_gguf(context, commands, &tensor, &bw->WD), err, cleanup_commands);
                }
            }
        }
    }

    vkllm_commands_free(context, commands);
    gguf_close(gguf);
    return VKLLM_ERR_OK;

cleanup_commands:
    if (commands)
        vkllm_commands_free(context, commands);
cleanup_gguf:
    if (gguf)
        gguf_close(gguf);
    if (model->weights.blocks)
    {
        for (size_t i = 0; i < model->weights.blocks->used_n; i++)
        {
            struct block_weights *bw = model->weights.blocks->data[i];
            if (bw)
                free(bw);
        }
        vkllm_array_block_weights_free(model->weights.blocks);
    }
    return err;
}

vkllm_err_t vkllm_models_llama2_free(struct vkllm_context *context, struct vkllm_models_llama2 *model)
{
    _CHECK_ARGS(context && model);

    if (model->graph)
    {
        vkllm_graph_free(context, model->graph);
    }

    if (model->meta.tokens)
    {
        for (size_t i = 0; i < model->meta.tokens->used_n; i++)
        {
            if (model->meta.tokens->data[i].text)
                free(model->meta.tokens->data[i].text);
        }
        vkllm_array_token_free(model->meta.tokens);
        model->meta.tokens = NULL;
    }

    if (model->weights.tok_embed_weights)
    {
        vkllm_tensor_free(context, model->weights.tok_embed_weights);
        model->weights.tok_embed_weights = NULL;
    }

    if (model->weights.output_norm_weight)
    {
        vkllm_tensor_free(context, model->weights.output_norm_weight);
        model->weights.output_norm_weight = NULL;
    }

    if (model->weights.output_weight)
    {
        vkllm_tensor_free(context, model->weights.output_weight);
        model->weights.output_weight = NULL;
    }

    if (model->weights.blocks)
    {
        for (size_t i = 0; i < model->weights.blocks->used_n; i++)
        {
            struct block_weights *bw = model->weights.blocks->data[i];
            if (!bw)
                continue;

            if (bw->attn_norm_weight)
                vkllm_tensor_free(context, bw->attn_norm_weight);
            if (bw->ffn_norm_weight)
                vkllm_tensor_free(context, bw->ffn_norm_weight);
            if (bw->WQ)
                vkllm_tensor_free(context, bw->WQ);
            if (bw->WK)
                vkllm_tensor_free(context, bw->WK);
            if (bw->WV)
                vkllm_tensor_free(context, bw->WV);
            if (bw->WO)
                vkllm_tensor_free(context, bw->WO);
            if (bw->WU)
                vkllm_tensor_free(context, bw->WU);
            if (bw->WG)
                vkllm_tensor_free(context, bw->WG);
            if (bw->WD)
                vkllm_tensor_free(context, bw->WD);

            free(bw);
        }
        vkllm_array_block_weights_free(model->weights.blocks);
        model->weights.blocks = NULL;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_models_llama2_build_model(struct vkllm_context *context, struct vkllm_models_llama2 *model,
                                            struct vkllm_tensor *input_toks)
{
    _CHECK_ARGS(context && model);
    _CHECK_ARGS(model->weights.tok_embed_weights && model->weights.output_norm_weight && model->weights.output_weight);
    _CHECK_ARGS(model->weights.blocks && model->weights.blocks->used_n > 0);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_graph *graph = NULL;
    struct vkllm_tensor *embedded = NULL;
    struct vkllm_tensor *hidden = NULL;

    _CHECK_JUMP(vkllm_graph_new(context, &graph), err, fail);

    uint32_t batch = input_toks->shapes[0];
    uint32_t seq_len = input_toks->shapes[3];
    uint32_t hidden_dim = model->meta.embedding_length;
    uint32_t vocab_size = model->meta.vocab_size;

    _CHECK_JUMP(vkllm_graph_add_input(context, graph, input_toks), err, fail_free_graph);

    uint32_t embed_shapes[4] = {batch, 1, seq_len, hidden_dim};
    struct vkllm_tensor *embed_srcs[] = {input_toks, model->weights.tok_embed_weights};
    uint32_t unk_tok = model->meta.padding_token_id;
    _CHECK_JUMP(vkllm_tensor_new(context, "embedded", embed_shapes, vkllm_dtype_float16, VKLLM_OP_EMBEDDING, embed_srcs,
                                 2, &unk_tok, sizeof(unk_tok), false, &embedded),
                err, fail_free_graph);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, embedded), err, fail_free_embedded);

    hidden = embedded;

    for (uint32_t i = 0; i < model->meta.block_count && i < model->weights.blocks->used_n; i++)
    {
        struct block_weights *bw = model->weights.blocks->data[i];
        if (!bw)
        {
            err = VKLLM_ERR_ARGS;
            goto fail_free_embedded;
        }

        struct vkllm_llama2_self_attn_layer_params attn_params = {
            .WK = bw->WK,
            .WQ = bw->WQ,
            .WV = bw->WV,
            .WO = bw->WO,
            .norm_weight = bw->attn_norm_weight,
            .norm_power = 2.0f,
            .norm_eps = model->meta.layer_norm_rms_epsilon,
            .freq_base = model->meta.rope_freq_base,
            .offsets = 0,
            .num_head = model->meta.head_count,
        };

        struct vkllm_llama2_ffn_layer_params ffn_params = {
            .WU = bw->WU,
            .WG = bw->WG,
            .WD = bw->WD,
            .norm_weight = bw->ffn_norm_weight,
            .norm_power = 2.0f,
            .norm_eps = model->meta.layer_norm_rms_epsilon,
        };

        struct vkllm_llama2_transformer_block_params block_params = {
            .attn = attn_params,
            .ffn = ffn_params,
        };

        char block_name[64];
        snprintf(block_name, sizeof(block_name), "block.%u", i);
        _CHECK_JUMP(vkllm_llama2_build_transformer_block(context, graph, hidden, block_params, block_name), err,
                    fail_free_embedded);
        hidden = graph->nodes->data[graph->nodes->used_n - 1];
    }

    struct vkllm_tensor *output_norm = NULL;
    struct vkllm_tensor *norm_srcs[] = {hidden, model->weights.output_norm_weight};
    struct vkllm_op_rmsnorm_params norm_params = {.power = 2.0f, .eps = model->meta.layer_norm_rms_epsilon};
    _CHECK_JUMP(vkllm_tensor_new(context, "output_norm", hidden->shapes, hidden->dtype, VKLLM_OP_RMSNORM, norm_srcs, 2,
                                 &norm_params, sizeof(norm_params), false, &output_norm),
                err, fail_free_embedded);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, output_norm), err, fail_free_output_norm);

    uint32_t logits_shapes[4] = {batch, 1, seq_len, vocab_size};
    struct vkllm_tensor *logits_srcs[] = {output_norm, model->weights.output_weight};
    struct vkllm_op_matmul_params matmul_params = {.scale = 1.0f, .act = 0};
    struct vkllm_tensor *logits = NULL;
    _CHECK_JUMP(vkllm_tensor_new(context, "logits", logits_shapes, output_norm->dtype, VKLLM_OP_MATMUL, logits_srcs, 2,
                                 &matmul_params, sizeof(matmul_params), false, &logits),
                err, fail_free_output_norm);
    _CHECK_JUMP(vkllm_graph_add_node(context, graph, logits), err, fail_free_logits);
    _CHECK_JUMP(vkllm_graph_set_output(context, graph, logits), err, fail_free_logits);

    model->graph = graph;

    return VKLLM_ERR_OK;

fail_free_logits:
    vkllm_tensor_free(context, logits);
fail_free_output_norm:
    vkllm_tensor_free(context, output_norm);
fail_free_embedded:
    vkllm_tensor_free(context, embedded);
fail_free_graph:
    vkllm_graph_free(context, graph);
fail:
    return err;
}
