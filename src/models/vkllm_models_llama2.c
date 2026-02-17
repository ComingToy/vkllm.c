#include "vkllm_models_llama2.h"
#include "../core/vkllm_commands.h"
#include "../core/vkllm_tensor.h"
#include "gguflib.h"
#include <log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vkllm_err_t parse_meta_kv(gguf_ctx *ctx, gguf_key *key, struct vkllm_models_llama2_weights *model)
{
    const char *name = key->name;
    size_t namelen = key->namelen;
    union gguf_value *val = key->val;
    uint32_t type = key->type;

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
    else if (strncmp(name, "llama.attention.head_count_kv", namelen) == 0)
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
        vkllm_tensor_new(context, name_buf, shapes, dtype, VKLLM_OP_COPY, NULL, 0, NULL, 0, false, ptensor);
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

vkllm_err_t vkllm_models_llama2_load_weights(struct vkllm_context *context, struct vkllm_models_llama2_weights *model,
                                             const char *file)
{
    _CHECK_ARGS(context && model && file);

    memset(model, 0, sizeof(*model));
    vkllm_err_t err = VKLLM_ERR_OK;

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
        parse_meta_kv(gguf, &key, model);
        printf("%.*s: [%s] ", (int)key.namelen, key.name, gguf_get_value_type_name(key.type));
        gguf_print_value(gguf, key.type, key.val, 0);
        printf("\n");
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

vkllm_err_t vkllm_models_llama2_free_weights(struct vkllm_context *context, struct vkllm_models_llama2_weights *model)
{
    _CHECK_ARGS(context && model);

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
