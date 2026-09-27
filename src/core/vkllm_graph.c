#include "vkllm_graph.h"
#include "src/core/vkllm_hashmap.h"
#include "src/core/vkllm_pipeline.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_context.h"
#include "vkllm_hashset.h"
#include "vkllm_tensor.h"

vkllm_err_t vkllm_graph_new(struct vkllm_context *context, struct vkllm_graph **graph)
{
    _CHECK_ARGS(context && graph);
    _NEW_AND_CHECK(*graph, struct vkllm_graph);
    struct vkllm_graph *p = *graph;

    vkllm_err_t err = VKLLM_ERR_OK;

    _CHECK_JUMP(vkllm_array_tensor_new(&p->input_nodes, 8), err, fail_new_inputs);
    _CHECK_JUMP(vkllm_array_tensor_new(&p->nodes, 128), err, fail_new_nodes);
    _CHECK_JUMP(vkllm_array_tensor_new(&p->external_nodes, 8), err, fail_new_external);
    p->output_node = NULL;

    _CHECK_JUMP(vkllm_commands_new(context, &p->commands), err, fail_new_commands);

    return err;

fail_new_commands:
    vkllm_array_tensor_free(p->external_nodes);
fail_new_external:
    vkllm_array_tensor_free(p->nodes);
fail_new_nodes:
    vkllm_array_tensor_free(p->input_nodes);
fail_new_inputs:
    free(p);
    return err;
}

vkllm_err_t vkllm_graph_add_input(struct vkllm_context *context, struct vkllm_graph *graph, struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && graph && tensor);

    _CHECK(vkllm_array_tensor_append(graph->input_nodes, tensor));
    _CHECK(vkllm_graph_add_node(context, graph, tensor));
    return VKLLM_ERR_OK;
}

// Ownership: appending transfers ownership of `tensor` to the graph. Once a
// tensor has been added (even if the append below fails), the graph is
// responsible for freeing it — callers must NOT free tensors after add_node.
vkllm_err_t vkllm_graph_add_node(struct vkllm_context *context, struct vkllm_graph *graph, struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && graph && tensor);

    vkllm_err_t err = vkllm_array_tensor_append(graph->nodes, tensor);
    if (err != VKLLM_ERR_OK)
    {
        // The array rejected the tensor; consume it here so ownership stays
        // single and callers never free a tensor they no longer own.
        vkllm_tensor_free(context, tensor);
    }

    return err;
}

// Register a tensor the graph must execute but does NOT own (e.g. KV cache
// tensors owned by the model). It participates in init/run traversal and is
// skipped by vkllm_graph_free.
vkllm_err_t vkllm_graph_add_external(struct vkllm_context *context, struct vkllm_graph *graph,
                                     struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && graph && tensor);
    vkllm_err_t err = vkllm_array_tensor_append(graph->external_nodes, tensor);
    if (err != VKLLM_ERR_OK)
    {
        log_error("add_external: append failed for %s (used=%zu alloc=%zu): %s", tensor->name,
                  graph->external_nodes->used_n, graph->external_nodes->alloc_n, vkllm_err_s(err));
        return err;
    }
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_graph_set_output(struct vkllm_context *context, struct vkllm_graph *graph,
                                   struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && graph && tensor);
    graph->output_node = tensor;
    return VKLLM_ERR_OK;
}

// Linear lookup by name; graphs are small (hundreds of nodes) and this is
// only used at build/prepare time, not on the hot path.
struct vkllm_tensor *vkllm_graph_find_node(struct vkllm_graph *graph, const char *name)
{
    if (!graph || !name)
        return NULL;

    for (size_t i = 0; i < graph->nodes->used_n; i++)
    {
        if (strncmp(graph->nodes->data[i]->name, name, sizeof(graph->nodes->data[i]->name)) == 0)
        {
            return graph->nodes->data[i];
        }
    }

    return NULL;
}

// Helper function to recursively initialize a tensor and its dependencies
static vkllm_err_t vkllm_graph_init_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                           struct vkllm_tensor *tensor, struct vkllm_hashset *visited)
{
    _CHECK_ARGS(context && commands && tensor && visited);

    // Check if already initialized using fast hash set lookup
    uint64_t tensor_key = (uint64_t)(uintptr_t)tensor;
    if (vkllm_hashset_contains(visited, tensor_key))
    {
        return VKLLM_ERR_OK; // Already visited
    }

    // First, recursively initialize all source tensors
    for (int i = 0; i < VKLLM_MAX_SRCS; i++)
    {
        if (tensor->srcs[i] != NULL)
        {
            _CHECK(vkllm_graph_init_tensor(context, commands, tensor->srcs[i], visited));
        }
    }

    // Mark this tensor as visited before initializing
    _CHECK(vkllm_hashset_insert(visited, tensor_key));

    // Dispatch through the op table; unknown ops have no entry.
    if (tensor->op < 0 || tensor->op >= VKLLM_OP_COUNTS)
    {
        log_error("unknown operation type: %d", (int)tensor->op);
        return VKLLM_ERR_ARGS;
    }

    const struct vkllm_op_desc *desc = &g_op_table[tensor->op];
    if (desc->init)
    {
        _CHECK(desc->init(context, commands, tensor));
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_graph_init(struct vkllm_context *context, struct vkllm_graph *graph)
{
    _CHECK_ARGS(context && graph && graph->output_node && graph->commands);

    // Create a hash set to track visited tensors for O(1) lookup
    struct vkllm_hashset *visited = NULL;
    vkllm_err_t err = vkllm_hashset_new(&visited, graph->nodes->used_n);
    if (err != VKLLM_ERR_OK)
    {
        return err;
    }

    // Recursively initialize starting from the output node
    err = vkllm_graph_init_tensor(context, graph->commands, graph->output_node, visited);

    // Initialize external (model-owned) tensors: KV cache update-rows ops.
    if (err == VKLLM_ERR_OK)
    {
        for (size_t i = 0; i < graph->external_nodes->used_n; i++)
        {
            err = vkllm_graph_init_tensor(context, graph->commands, graph->external_nodes->data[i], visited);
            if (err != VKLLM_ERR_OK)
                break;
        }
    }

    // Clean up the visited hash set
    vkllm_hashset_free(visited);

    return err;
}

// Helper function to recursively run a tensor and its dependencies
static vkllm_err_t vkllm_graph_run_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor, struct vkllm_hashset *visited)
{
    _CHECK_ARGS(context && commands && tensor && visited);

    // Check if already executed using fast hash set lookup
    uint64_t tensor_key = (uint64_t)(uintptr_t)tensor;
    if (vkllm_hashset_contains(visited, tensor_key))
    {
        return VKLLM_ERR_OK; // Already executed
    }

    // First, recursively execute all source tensors (dependencies)
    for (int i = 0; i < VKLLM_MAX_SRCS; i++)
    {
        if (tensor->srcs[i] != NULL)
        {
            _CHECK(vkllm_graph_run_tensor(context, commands, tensor->srcs[i], visited));
        }
    }

    // Mark this tensor as visited before execution
    _CHECK(vkllm_hashset_insert(visited, tensor_key));

    // Dispatch through the op table; unknown ops have no entry.
    if (tensor->op < 0 || tensor->op >= VKLLM_OP_COUNTS)
    {
        log_error("unknown operation type: %d", (int)tensor->op);
        return VKLLM_ERR_ARGS;
    }

    const struct vkllm_op_desc *desc = &g_op_table[tensor->op];
    if (desc->run)
    {
        _CHECK(desc->run(context, commands, tensor));
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_graph_run(struct vkllm_context *context, struct vkllm_graph *graph)
{
    _CHECK_ARGS(context && graph && graph->output_node && graph->commands);
    _CHECK(vkllm_commands_begin(context, graph->commands));

    // Create a hash set to track visited tensors for O(1) lookup
    struct vkllm_hashset *visited = NULL;
    vkllm_err_t err = vkllm_hashset_new(&visited, graph->nodes->used_n);
    if (err != VKLLM_ERR_OK)
    {
        return err;
    }

    // Recursively execute starting from the output node
    err = vkllm_graph_run_tensor(context, graph->commands, graph->output_node, visited);

    // Execute external (model-owned) tensors: KV cache update-rows writes.
    if (err == VKLLM_ERR_OK)
    {
        for (size_t i = 0; i < graph->external_nodes->used_n; i++)
        {
            err = vkllm_graph_run_tensor(context, graph->commands, graph->external_nodes->data[i], visited);
            if (err != VKLLM_ERR_OK)
                break;
        }
    }

    // Clean up the visited hash set
    vkllm_hashset_free(visited);

    _CHECK(vkllm_commands_end(context, graph->commands));
    _CHECK(vkllm_commands_submit(context, graph->commands));
    _CHECK(vkllm_commands_wait_exec(context, graph->commands));
    return err;
}

// Helper function to recursively post-run a tensor and its dependencies
static vkllm_err_t vkllm_graph_post_run_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                               struct vkllm_tensor *tensor, struct vkllm_hashset *visited)
{
    _CHECK_ARGS(context && commands && tensor && visited);

    // Check if already post-run using fast hash set lookup
    uint64_t tensor_key = (uint64_t)(uintptr_t)tensor;
    if (vkllm_hashset_contains(visited, tensor_key))
    {
        return VKLLM_ERR_OK; // Already post-run
    }

    // First, recursively post-run all source tensors
    for (int i = 0; i < VKLLM_MAX_SRCS; i++)
    {
        if (tensor->srcs[i] != NULL)
        {
            _CHECK(vkllm_graph_post_run_tensor(context, commands, tensor->srcs[i], visited));
        }
    }

    // Mark this tensor as visited before post-run
    _CHECK(vkllm_hashset_insert(visited, tensor_key));

    // Dispatch through the op table; unknown ops have no entry.
    if (tensor->op < 0 || tensor->op >= VKLLM_OP_COUNTS)
    {
        log_error("unknown operation type: %d", (int)tensor->op);
        return VKLLM_ERR_ARGS;
    }

    const struct vkllm_op_desc *desc = &g_op_table[tensor->op];
    if (desc->post_run)
    {
        _CHECK(desc->post_run(context, commands, tensor));
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_graph_post_run(struct vkllm_context *context, struct vkllm_graph *graph)
{
    _CHECK_ARGS(context && graph && graph->output_node && graph->commands);

    // Create a hash set to track visited tensors for O(1) lookup
    struct vkllm_hashset *visited = NULL;
    vkllm_err_t err = vkllm_hashset_new(&visited, graph->nodes->used_n);
    if (err != VKLLM_ERR_OK)
    {
        return err;
    }

    // Recursively post-run starting from the output node
    err = vkllm_graph_post_run_tensor(context, graph->commands, graph->output_node, visited);

    // Post-run external (model-owned) tensors: KV cache update-rows ops.
    if (err == VKLLM_ERR_OK)
    {
        for (size_t i = 0; i < graph->external_nodes->used_n; i++)
        {
            err = vkllm_graph_post_run_tensor(context, graph->commands, graph->external_nodes->data[i], visited);
            if (err != VKLLM_ERR_OK)
                break;
        }
    }

    // Clean up the visited hash set
    vkllm_hashset_free(visited);

    return err;
}

vkllm_err_t vkllm_graph_free(struct vkllm_context *context, struct vkllm_graph *graph)
{
    _CHECK_ARGS(context && graph);

    for (size_t i = 0; i < graph->nodes->used_n; i++)
    {
        vkllm_tensor_free(context, graph->nodes->data[i]);
    }

    vkllm_array_tensor_free(graph->nodes);
    vkllm_array_tensor_free(graph->input_nodes);
    vkllm_array_tensor_free(graph->external_nodes);
    vkllm_commands_free(context, graph->commands);
    free(graph);

    return VKLLM_ERR_OK;
}
