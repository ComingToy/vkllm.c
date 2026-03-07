#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_errors.h"
#include "src/core/vkllm_graph.h"
#include "src/core/vkllm_hashmap.h"
#include "src/core/vkllm_ops.h"
#include "src/core/vkllm_pipeline.h"
#include "src/core/vkllm_tensor.h"
#include "src/models/vkllm_models_llama2.h"
#include "tests/vkllm_test_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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

static uint32_t extract_output_tok(struct vkllm_graph *graph)
{
    struct vkllm_tensor *node = graph->output_node;
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

    return max_index;
}

static int hashmap_entry_cmp(const void *lhs, const void *rhs)
{
    return ((struct vkllm_hashmap_entry *)lhs)->value - ((struct vkllm_hashmap_entry *)rhs)->value;
}

int main(const int argc, const char *argv[])
{
    if (argc != 3)
    {
        log_error("usage: %s <path to gguf> <num_tokens>", argv[0]);
        return -1;
    }

    uint32_t num_tokens = atoi(argv[2]);
    if (num_tokens == 0)
    {
        log_error("invalid num_tokens: %s", argv[2]);
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
        goto cleanup_model;
    }

    fprintf(stderr, "input ids: ");
    for (uint32_t i = 0; i < token_ids->used_n; ++i)
    {
        fprintf(stderr, "%u ", token_ids->data[i]);
    }

    uint32_t offset = 0;
    struct vkllm_graph *graph = NULL;
    struct vkllm_array_token_id *output_tokens = NULL;
    vkllm_array_token_id_new(&output_tokens, 32);

    double total_time_ms = 0.0;
    struct timeval tv_start, tv_end;

    for (uint32_t i = 0; i < num_tokens; ++i)
    {
        struct vkllm_tensor *input_toks = NULL;
        uint32_t input_shapes[] = {1, 1, 1, token_ids->used_n};

        _CHECK_JUMP(vkllm_tensor_new(context, "input_toks", input_shapes, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0,
                                     NULL, 0, true, &input_toks),
                    err, cleanup_token_ids);
        _CHECK_JUMP(vkllm_graph_new(context, &graph), err, cleanup_token_ids);

        _CHECK_JUMP(vkllm_models_llama2_build_graph(context, &model, input_toks, graph, offset), err, cleanup_graph);
        _CHECK_JUMP(vkllm_graph_init(context, graph), err, cleanup_graph);
        _CHECK_JUMP(vkllm_commands_upload(context, graph->commands, input_toks, (const uint8_t *)token_ids->data,
                                          sizeof(uint32_t) * token_ids->used_n),
                    err, cleanup_graph);

        if (i == 1)
        {
            gettimeofday(&tv_start, NULL);
        }

        _CHECK_JUMP(vkllm_graph_run(context, graph), err, cleanup_graph);
        _CHECK_JUMP(vkllm_graph_post_run(context, graph), err, cleanup_graph);

        if (i != 0)
        {
            for (uint32_t i = 0; i < graph->nodes->used_n; ++i)
            {
                struct vkllm_tensor *node = graph->nodes->data[i];
                struct vkllm_pipeline *pipeline = node->pipeline;
                if (!pipeline)
                    continue;

                uint64_t cost = 0;
                vkllm_pipeline_query_exec_time(context, pipeline, &cost);
                context->stats.op_time_costs[node->op] += cost;

                uint64_t acc = 0;
                vkllm_hashmap_get(context->stats.node_time_cost, node->name, &acc);
                acc += cost;
                vkllm_hashmap_insert(context->stats.node_time_cost, node->name, acc);
            }
        }
#if 0
        if (i == 0)
        {
            for (uint32_t i = 0; i < graph->nodes->used_n; ++i)
            {
                struct vkllm_tensor *node = graph->nodes->data[i];
                log_info("node %s use pipeline %s", node->name, node->pipeline ? node->pipeline->name : "NULL");
            }
            for (uint32_t i = 0; i < graph->nodes->used_n; ++i)
            {
                struct vkllm_tensor *node = graph->nodes->data[i];
                if (strcmp(node->name, "block.0.attn.norm") == 0 || strcmp(node->name, "block.0.attn.RQ") == 0 ||
                    strcmp("embedded", node->name) == 0 || strcmp("block.0.attn.Q_ref", node->name) == 0)
                {
                    print_tensor_mean(context, graph->commands, node);
                    uint32_t seq_len = node->shapes[2], num_head = model.meta.head_count,
                             hiddden = node->shapes[3] / model.meta.head_count;
                    uint32_t shape[] = {1, seq_len, num_head, hiddden};
                    // _CHECK_JUMP(vkllm_tensor_reshape(context, node, shape), err, cleanup_model);
                    log_info("tensor %s shape = [%u, %u, %u, %u]", node->name, node->shapes[0], node->shapes[1],
                             node->shapes[2], node->shapes[3]);
                    print_first_n(context, graph->commands, node, 0, 0, 0, 32);
                    print_first_n(context, graph->commands, node, 0, 0, 1, 32);
                    // print_first_n(context, model.graph->commands, node, 0, 0, 1, 100);
                }
            }

            {
                struct vkllm_tensor *node = model.weights.blocks->data[0]->WQ;
                print_tensor_mean(context, graph->commands, node);
                print_first_n(context, graph->commands, node, 0, 0, 0, 64);
                print_first_n(context, graph->commands, node, 0, 0, 1, 64);
            }
        }
#endif

        vkllm_tensor_invalid_cache(context, graph->output_node);
        uint32_t pred_tok = extract_output_tok(graph);

        offset += token_ids->used_n;
        vkllm_array_token_id_append(output_tokens, pred_tok);
        token_ids->used_n = 0;
        vkllm_array_token_id_append(token_ids, pred_tok);
        vkllm_graph_free(context, graph);
        graph = NULL;
    }

    gettimeofday(&tv_end, NULL);
    total_time_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
    double tokens_per_second = ((num_tokens - 1.0) / total_time_ms) * 1000.0;
    log_info("Average tokens/second: %.2f (generated %u tokens in %.2f ms)", tokens_per_second, num_tokens - 1,
             total_time_ms);

    log_info("genereate sentence: ");
    for (uint32_t i = 0; i < output_tokens->used_n; ++i)
    {
        uint32_t pred_tok = output_tokens->data[i];
        // log_info("pred tok: %u, piece: %s", pred_tok, model.meta.tokens->data[pred_tok].text);
        fprintf(stdout, "%s", model.meta.tokens->data[pred_tok].text);
    }

    for (uint32_t i = 0; i < VKLLM_OP_COUNTS; ++i)
    {
        fprintf(stderr, "op %s total time cost: %lums\n", vkllm_op_s(i),
                (unsigned long)context->stats.op_time_costs[i] / 1000000);
    }

    struct vkllm_hashmap_entry *entries =
        malloc(sizeof(struct vkllm_hashmap_entry) * context->stats.node_time_cost->capacity);

    uint32_t wi = 0;
    for (uint32_t i = 0; i < context->stats.node_time_cost->capacity; ++i)
    {
        struct vkllm_hashmap_entry entry = context->stats.node_time_cost->entries[i];
        if (entry.occupied)
        {
            entries[wi++] = entry;
        }
    }

    qsort(entries, wi, sizeof(struct vkllm_hashmap_entry), hashmap_entry_cmp);
    for (uint32_t i = 0; i < wi; ++i)
    {
        fprintf(stderr, "node %s total cost: %lu\n", entries[i].key, entries[i].value/(1000*1000));
    }

cleanup_graph:
    if (graph)
        vkllm_graph_free(context, graph);
cleanup_token_ids:
    vkllm_array_token_id_free(token_ids);
cleanup_model:
    vkllm_models_llama2_free(context, &model);
cleanup_context:
    vkllm_context_free(context);
    return err;
}
