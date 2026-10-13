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
#include <inttypes.h>
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

    if (node->dtype != vkllm_dtype_uint32)
    {
        log_error("output node must be uint32 type");
        return 0;
    }

    uint32_t *data = (uint32_t *)node->data.host;
    uint32_t strides[] = {node->strides[0] / 4, node->strides[1] / 4, node->strides[2] / 4, node->strides[3] / 4};

    uint32_t i =
        (node->shapes[0] - 1) * strides[0] + (node->shapes[1] - 1) * strides[1] + (node->shapes[2] - 1) * strides[2];
    return data[i];
}

static int hashmap_entry_cmp(const void *lhs, const void *rhs)
{
    return ((struct vkllm_hashmap_entry *)lhs)->value - ((struct vkllm_hashmap_entry *)rhs)->value;
}

int main(const int argc, const char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        log_error("usage: %s <path to gguf> <num_tokens> [prompt]", argv[0]);
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
    // Prompt argument: literal text, or "@file" to read from a file.
    char *prompt = NULL;
    if (argc > 3)
    {
        if (argv[3][0] == '@')
        {
            FILE *f = fopen(argv[3] + 1, "r");
            if (!f)
            {
                log_error("cannot open prompt file: %s", argv[3] + 1);
                return -1;
            }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            prompt = (char *)malloc(sz + 1);
            size_t rd = fread(prompt, 1, sz, f);
            prompt[rd] = '\0';
            fclose(f);
        }
        else
        {
            prompt = strdup(argv[3]);
        }
    }
    else
    {
        prompt = strdup("Q: What is the largest animal? A:");
    }

    err = vkllm_models_llama2_tokenize(&model, prompt, &token_ids);
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
    struct vkllm_array_token_id *output_tokens = NULL;
    vkllm_array_token_id_new(&output_tokens, 32);

    double total_time_ms = 0.0;
    struct timeval tv_start, tv_end;
    struct vkllm_graph *graph = NULL;
    struct vkllm_tensor *input_toks = NULL;

    // Prefill: one throwaway graph over the whole prompt (seq = n_prompt).
    {
        uint32_t input_shapes[] = {1, 1, 1, token_ids->used_n};
        _CHECK_JUMP(vkllm_tensor_new(context, "input_toks", input_shapes, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0,
                                     NULL, 0, true, &input_toks),
                    err, cleanup_token_ids);
        _CHECK_JUMP(vkllm_graph_new(context, &graph), err, cleanup_token_ids);
        _CHECK_JUMP(vkllm_models_llama2_build_graph(context, &model, input_toks, graph, 0), err, cleanup_graph);
        _CHECK_JUMP(vkllm_graph_init(context, graph), err, cleanup_graph);
        _CHECK_JUMP(vkllm_commands_upload(context, graph->commands, input_toks, (const uint8_t *)token_ids->data,
                                          sizeof(uint32_t) * token_ids->used_n),
                    err, cleanup_graph);
        struct timeval pf_start, pf_end;
        gettimeofday(&pf_start, NULL);
        _CHECK_JUMP(vkllm_graph_run(context, graph), err, cleanup_graph);
        _CHECK_JUMP(vkllm_graph_post_run(context, graph), err, cleanup_graph);
        gettimeofday(&pf_end, NULL);
        double prefill_ms =
            (pf_end.tv_sec - pf_start.tv_sec) * 1000.0 + (pf_end.tv_usec - pf_start.tv_usec) / 1000.0;
        log_info("PREFILL: %u tokens in %.2f ms (%.1f tok/s)", token_ids->used_n, prefill_ms,
                 token_ids->used_n / prefill_ms * 1000.0);

        vkllm_tensor_invalid_cache(context, graph->output_node);
        uint32_t pred_tok = extract_output_tok(graph);
        vkllm_array_token_id_append(output_tokens, pred_tok);

        offset = token_ids->used_n;
        token_ids->used_n = 0;
        vkllm_array_token_id_append(token_ids, pred_tok);

        vkllm_graph_free(context, graph);
        graph = NULL;
    }

    const uint64_t allocs_before = context->stats.tensor_alloc_counts;
    const uint64_t frees_before = context->stats.tensor_free_counts;

    // Decode: build the seq=1 graph once, then re-run it every step.
    uint32_t input_shapes[] = {1, 1, 1, 1};
    _CHECK_JUMP(vkllm_tensor_new(context, "input_toks", input_shapes, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0,
                                 NULL, 0, true, &input_toks),
                err, cleanup_token_ids);
    _CHECK_JUMP(vkllm_graph_new(context, &graph), err, cleanup_token_ids);
    _CHECK_JUMP(vkllm_models_llama2_build_graph(context, &model, input_toks, graph, offset), err, cleanup_graph);
    _CHECK_JUMP(vkllm_graph_init(context, graph), err, cleanup_graph);

    // Optional warmup steps discarded from the timing window (VKLLM_WARMUP=N)
    // — protocol parity with llama-bench, whose explicit warmup absorbs the
    // GPU clock ramp (first ~100 steps run ~12% slow on Apple Silicon).
    // Aligned mode (VKLLM_ALIGNED=1): skip the logits readback + CPU argmax
    // and feed a fixed next token, matching llama-bench's test_gen (random
    // next token, logits never read). Isolates pure model-execution speed.
    const int aligned = getenv("VKLLM_ALIGNED") != NULL;
    const char *warm_env = getenv("VKLLM_WARMUP");
    uint32_t warmup = warm_env ? (uint32_t)atoi(warm_env) : 0;
    // Per-phase host timing (VKLLM_PHASE_PROFILE): isolates the host-side
    // work llama-bench does NOT pay (logits readback, prepare_step's linear
    // node lookups) from the GPU-bound graph execution.
    const int phase_prof = getenv("VKLLM_PHASE_PROFILE") != NULL;
    double t_prep = 0, t_run = 0, t_post = 0, t_read = 0;
    struct timeval pa, pb;
    for (uint32_t i = 1; i < num_tokens + warmup; ++i)
    {
        const bool in_window = (i > warmup);
        if (phase_prof && in_window)
            gettimeofday(&pa, NULL);
        _CHECK_JUMP(vkllm_models_llama2_prepare_step(context, &model, graph, offset, 1), err, cleanup_graph);
        _CHECK_JUMP(vkllm_commands_upload(context, graph->commands, input_toks, (const uint8_t *)token_ids->data,
                                          sizeof(uint32_t)),
                    err, cleanup_graph);
        if (phase_prof && in_window)
        {
            gettimeofday(&pb, NULL);
            t_prep += (pb.tv_sec - pa.tv_sec) * 1000.0 + (pb.tv_usec - pa.tv_usec) / 1000.0;
            gettimeofday(&pa, NULL);
        }

        if (i == 1 + warmup)
        {
            gettimeofday(&tv_start, NULL);
        }

        _CHECK_JUMP(vkllm_graph_run(context, graph), err, cleanup_graph);
        if (phase_prof && in_window)
        {
            gettimeofday(&pb, NULL);
            t_run += (pb.tv_sec - pa.tv_sec) * 1000.0 + (pb.tv_usec - pa.tv_usec) / 1000.0;
            gettimeofday(&pa, NULL);
        }
        _CHECK_JUMP(vkllm_graph_post_run(context, graph), err, cleanup_graph);

        uint32_t pred_tok = 5; // fixed token (aligned mode: no readback)
        if (!aligned)
        {
            vkllm_tensor_invalid_cache(context, graph->output_node);
            pred_tok = extract_output_tok(graph);
        }
        if (phase_prof && in_window)
        {
            gettimeofday(&pb, NULL);
            t_post += (pb.tv_sec - pa.tv_sec) * 1000.0 + (pb.tv_usec - pa.tv_usec) / 1000.0;
        }

        offset += 1;
        vkllm_array_token_id_append(output_tokens, pred_tok);
        token_ids->used_n = 0;
        vkllm_array_token_id_append(token_ids, pred_tok);
    }

    gettimeofday(&tv_end, NULL);
    log_info("decode window: %" PRIu64 " tensor allocs, %" PRIu64 " tensor frees over %u steps",
             context->stats.tensor_alloc_counts - allocs_before,
             context->stats.tensor_free_counts - frees_before,
             num_tokens - 1);
    total_time_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
    if (phase_prof)
    {
        uint32_t n = num_tokens - 1;
        log_info("phase profile: prepare+upload %.3f ms/step | graph_run %.3f | post+read %.3f",
                 t_prep / n, t_run / n, t_post / n);
    }
    double tokens_per_second = ((num_tokens - 1.0) / total_time_ms) * 1000.0;
    log_info("Average tokens/second: %.2f (generated %u tokens in %.2f ms, %u warmup steps discarded)",
             tokens_per_second, num_tokens - 1, total_time_ms, warmup);

    log_info("genereate sentence: ");
    for (uint32_t i = 0; i < output_tokens->used_n; ++i)
    {
        uint32_t pred_tok = output_tokens->data[i];
        // log_info("pred tok: %u, piece: %s", pred_tok, model.meta.tokens->data[pred_tok].text);
        fprintf(stdout, "%s", model.meta.tokens->data[pred_tok].text);
    }

#if _LOG_NODE_TIME_COST
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
        fprintf(stderr, "node %s total cost: %lu\n", entries[i].key, entries[i].value / (1000 * 1000));
    }
#endif

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
