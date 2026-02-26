#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_errors.h"
#include "src/core/vkllm_graph.h"
#include "src/core/vkllm_tensor.h"
#include "src/models/vkllm_models_llama2.h"
#include "tests/vkllm_test_common.h"
#include <string.h>

static vkllm_err_t print_first_n(struct vkllm_context *context, struct vkllm_commands *commands,
                                 struct vkllm_tensor *tensor, uint32_t b, uint32_t c, uint32_t h, uint32_t n)
{
    if (!tensor->data.mapped)
    {
        return VKLLM_ERR_ARGS;
    }

    uint8_t *data = (uint8_t *)tensor->data.host;
    uint32_t count = n < tensor->shapes[3] ? n : tensor->shapes[3];

    fprintf(stderr, "%s [%u,%u,%u,:%u]: ", tensor->name, b, c, h, count);

    switch (tensor->dtype)
    {
    case vkllm_dtype_float32: {
        float *base = (float *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(float);
        uint32_t stride1 = tensor->strides[1] / sizeof(float);
        uint32_t stride2 = tensor->strides[2] / sizeof(float);
        uint32_t stride3 = tensor->strides[3] / sizeof(float);
        float *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%f ", p[w * stride3]);
        break;
    }
    case vkllm_dtype_float16: {
        vkllm_fp16_pack *base = (vkllm_fp16_pack *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(vkllm_fp16_pack);
        uint32_t stride1 = tensor->strides[1] / sizeof(vkllm_fp16_pack);
        uint32_t stride2 = tensor->strides[2] / sizeof(vkllm_fp16_pack);
        uint32_t stride3 = tensor->strides[3] / sizeof(vkllm_fp16_pack);
        vkllm_fp16_pack *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%f ", vkllm_fp16_to_fp32(p[w * stride3]));
        break;
    }
    case vkllm_dtype_int8: {
        int8_t *base = (int8_t *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(int8_t);
        uint32_t stride1 = tensor->strides[1] / sizeof(int8_t);
        uint32_t stride2 = tensor->strides[2] / sizeof(int8_t);
        uint32_t stride3 = tensor->strides[3] / sizeof(int8_t);
        int8_t *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%d ", p[w * stride3]);
        break;
    }
    case vkllm_dtype_uint32: {
        uint32_t *base = (uint32_t *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(uint32_t);
        uint32_t stride1 = tensor->strides[1] / sizeof(uint32_t);
        uint32_t stride2 = tensor->strides[2] / sizeof(uint32_t);
        uint32_t stride3 = tensor->strides[3] / sizeof(uint32_t);
        uint32_t *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%u ", p[w * stride3]);
        break;
    }
    default:
        return VKLLM_ERR_ARGS;
    }

    fprintf(stderr, "\n");
    return VKLLM_ERR_OK;
}

static vkllm_err_t print_tensor_mean(struct vkllm_context *context, struct vkllm_commands *commands,
                                     struct vkllm_tensor *tensor)
{
    uint8_t *data;
    bool need_free = false;
    data = (uint8_t *)tensor->data.host;

    if (!tensor->data.mapped)
    {
        return VKLLM_ERR_ARGS;
    }

    uint32_t total = tensor->shapes[0] * tensor->shapes[1] * tensor->shapes[2] * tensor->shapes[3];
    double sum = 0.0;

    switch (tensor->dtype)
    {
    case vkllm_dtype_float32: {
        float *base = (float *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(float);
        uint32_t stride1 = tensor->strides[1] / sizeof(float);
        uint32_t stride2 = tensor->strides[2] / sizeof(float);
        uint32_t stride3 = tensor->strides[3] / sizeof(float);
        for (uint32_t n = 0; n < tensor->shapes[0]; ++n)
            for (uint32_t c = 0; c < tensor->shapes[1]; ++c)
                for (uint32_t h = 0; h < tensor->shapes[2]; ++h)
                    for (uint32_t w = 0; w < tensor->shapes[3]; ++w)
                        sum += base[n * stride0 + c * stride1 + h * stride2 + w * stride3];
        break;
    }
    case vkllm_dtype_float16: {
        vkllm_fp16_pack *base = (vkllm_fp16_pack *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(vkllm_fp16_pack);
        uint32_t stride1 = tensor->strides[1] / sizeof(vkllm_fp16_pack);
        uint32_t stride2 = tensor->strides[2] / sizeof(vkllm_fp16_pack);
        uint32_t stride3 = tensor->strides[3] / sizeof(vkllm_fp16_pack);
        for (uint32_t n = 0; n < tensor->shapes[0]; ++n)
            for (uint32_t c = 0; c < tensor->shapes[1]; ++c)
                for (uint32_t h = 0; h < tensor->shapes[2]; ++h)
                    for (uint32_t w = 0; w < tensor->shapes[3]; ++w)
                        sum += vkllm_fp16_to_fp32(base[n * stride0 + c * stride1 + h * stride2 + w * stride3]);
        break;
    }
    case vkllm_dtype_int8: {
        int8_t *base = (int8_t *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(int8_t);
        uint32_t stride1 = tensor->strides[1] / sizeof(int8_t);
        uint32_t stride2 = tensor->strides[2] / sizeof(int8_t);
        uint32_t stride3 = tensor->strides[3] / sizeof(int8_t);
        for (uint32_t n = 0; n < tensor->shapes[0]; ++n)
            for (uint32_t c = 0; c < tensor->shapes[1]; ++c)
                for (uint32_t h = 0; h < tensor->shapes[2]; ++h)
                    for (uint32_t w = 0; w < tensor->shapes[3]; ++w)
                        sum += base[n * stride0 + c * stride1 + h * stride2 + w * stride3];
        break;
    }
    case vkllm_dtype_uint32: {
        uint32_t *base = (uint32_t *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(uint32_t);
        uint32_t stride1 = tensor->strides[1] / sizeof(uint32_t);
        uint32_t stride2 = tensor->strides[2] / sizeof(uint32_t);
        uint32_t stride3 = tensor->strides[3] / sizeof(uint32_t);
        for (uint32_t n = 0; n < tensor->shapes[0]; ++n)
            for (uint32_t c = 0; c < tensor->shapes[1]; ++c)
                for (uint32_t h = 0; h < tensor->shapes[2]; ++h)
                    for (uint32_t w = 0; w < tensor->shapes[3]; ++w)
                        sum += base[n * stride0 + c * stride1 + h * stride2 + w * stride3];
        break;
    }
    default:
        if (need_free)
            free(data);
        return VKLLM_ERR_ARGS;
    }

    log_info("tensor %s mean: %f", tensor->name, sum / total);

    if (need_free)
        free(data);

    return VKLLM_ERR_OK;
}

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

    for (uint32_t i = 0; i < model.graph->nodes->used_n; ++i)
    {
        struct vkllm_tensor *node = model.graph->nodes->data[i];
        if (strcmp(node->name, "block.0.attn.Q") == 0 || strcmp(node->name, "block.0.attn.K") == 0 ||
            strcmp(node->name, "block.0.attn.V") == 0 || strcmp(node->name, "block.0.attn.norm") == 0)
        {
            print_tensor_mean(context, model.graph->commands, node);
            print_first_n(context, model.graph->commands, node, 0, 0, 0, 64);
            print_first_n(context, model.graph->commands, node, 0, 0, 1, 64);
        }
    }

    {
        struct vkllm_tensor *node = model.weights.blocks->data[0]->WQ;
        print_tensor_mean(context, model.graph->commands, node);
        print_first_n(context, model.graph->commands, node, 0, 0, 0, 64);
        print_first_n(context, model.graph->commands, node, 0, 0, 1, 64);
    }

    uint32_t pred_tok = extract_output_tok(&model);
    log_info("pred tok: %u", pred_tok);

cleanup_model:
    vkllm_models_llama2_free(context, &model);
cleanup_context:
    vkllm_context_free(context);
    return err;
}
