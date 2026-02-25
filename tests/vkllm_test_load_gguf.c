#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_errors.h"
#include "src/core/vkllm_graph.h"
#include "src/core/vkllm_tensor.h"
#include "src/models/vkllm_models_llama2.h"

static uint32_t extract_output_tok(struct vkllm_models_llama2 *model)
{
    struct vkllm_tensor *node = model->graph->output_node;
    if (!node->data.mapped)
    {
        return 0;
    }

    vkllm_fp16_pack *data = (vkllm_fp16_pack *)node->data.host;
    uint32_t strides[] = {node->strides[0] / 2, node->strides[1] / 2, node->strides[2] / 2, node->strides[3] / 2};

    vkllm_fp16_pack *p = data + (node->shapes[2] - 1) * strides[2];

    float max_logits = vkllm_fp16_to_fp32(*p);
    uint32_t max_index = 0;

    for (uint32_t i = 0; i < node->shapes[3]; ++i)
    {
        float val = vkllm_fp16_to_fp32(p[i]);
        if (val > max_logits)
        {
            max_index = i;
            max_logits = val;
        }
    }

    log_info("max logits: %f, pred tok: %u", max_logits, max_index);
    return max_index;
}

int main(const int argc, const char *argv[])
{
    if (argc != 2)
    {
        log_error("usage: %s <path to gguf>", argv[0]);
        return -1;
    }

    struct vkllm_context *context = NULL;
    struct vkllm_models_llama2 model;

    vkllm_err_t err = vkllm_context_new(0, &context);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at creating context: %s", vkllm_err_s(err));
        return -1;
    }

    err = vkllm_models_llama2_load(context, &model, argv[1]);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at loading weights: %s", vkllm_err_s(err));
        goto cleanup_context;
    }

    struct vkllm_array_token_id *token_ids;
    err = vkllm_models_llama2_tokenize(&model, "Q: What is the largest animal? A:", &token_ids);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at tokenize: %s", vkllm_err_s(err));
        goto cleanup_context;
    }

    fprintf(stderr, "input ids: ");
    for (uint32_t i = 0; i < token_ids->used_n; ++i)
    {
        fprintf(stderr, "%u ", token_ids->data[i]);
    }

    struct vkllm_tensor *input_toks = NULL;
    uint32_t input_shapes[] = {1, 1, 1, token_ids->used_n};

    _CHECK_JUMP(vkllm_tensor_new(context, "input_toks", input_shapes, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0, NULL,
                                 0, true, &input_toks),
                err, cleanup_context);

    _CHECK_JUMP(vkllm_models_llama2_build_model(context, &model, input_toks), err, cleanup_model);
    _CHECK_JUMP(vkllm_graph_init(context, model.graph), err, cleanup_model);
    _CHECK_JUMP(vkllm_commands_upload(context, model.graph->commands, input_toks, (const uint8_t *)token_ids->data,
                                      sizeof(uint32_t) * token_ids->used_n),
                err, cleanup_model);
    _CHECK_JUMP(vkllm_graph_run(context, model.graph), err, cleanup_model);
    _CHECK_JUMP(vkllm_graph_post_run(context, model.graph), err, cleanup_model);

    uint32_t pred_tok = extract_output_tok(&model);
    log_info("pred tok: %u", pred_tok);

cleanup_model:
    vkllm_models_llama2_free(context, &model);
cleanup_context:
    vkllm_context_free(context);
    return err;
}
