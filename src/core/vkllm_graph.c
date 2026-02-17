#include "vkllm_graph.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_context.h"
#include "vkllm_hashset.h"
#include "vkllm_op_bin.h"
#include "vkllm_op_copy.h"
#include "vkllm_op_embedding.h"
#include "vkllm_op_matmul.h"
#include "vkllm_op_rmsnorm.h"
#include "vkllm_op_rope.h"
#include "vkllm_op_softmax.h"
#include "vkllm_tensor.h"

vkllm_err_t vkllm_graph_new(struct vkllm_context *context, struct vkllm_graph **graph)
{
    _CHECK_ARGS(context && graph);
    _NEW_AND_CHECK(*graph, struct vkllm_graph);
    struct vkllm_graph *p = *graph;

    vkllm_err_t err = VKLLM_ERR_OK;

    _CHECK_JUMP(vkllm_array_tensor_new(&p->input_nodes, 8), err, fail_new_inputs);
    _CHECK_JUMP(vkllm_array_tensor_new(&p->nodes, 128), err, fail_new_nodes);
    p->output_node = NULL;

    _CHECK_JUMP(vkllm_commands_new(context, &p->commands), err, fail_new_commands);

    return err;

fail_new_commands:
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

vkllm_err_t vkllm_graph_add_node(struct vkllm_context *context, struct vkllm_graph *graph, struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && graph && tensor);

    _CHECK(vkllm_array_tensor_append(graph->nodes, tensor));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_graph_set_output(struct vkllm_context *context, struct vkllm_graph *graph,
                                   struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && graph && tensor);
    graph->output_node = tensor;
    return VKLLM_ERR_OK;
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

    // Initialize this tensor based on its operation type
    switch (tensor->op)
    {
    case VKLLM_OP_NONE:
        // Input tensors or constants, no initialization needed
        break;
    case VKLLM_OP_COPY:
        _CHECK(vkllm_op_copy_init(context, commands, tensor));
        break;
    case VKLLM_OP_BIN:
        _CHECK(vkllm_op_bin_init(context, commands, tensor));
        break;
    case VKLLM_OP_EMBEDDING:
        _CHECK(vkllm_op_embedding_init(context, commands, tensor));
        break;
    case VKLLM_OP_RMSNORM:
        _CHECK(vkllm_op_rmsnorm_init(context, commands, tensor));
        break;
    case VKLLM_OP_MATMUL:
        _CHECK(vkllm_op_matmul_init(context, commands, tensor));
        break;
    case VKLLM_OP_ROPE:
        _CHECK(vkllm_op_rope_init(context, commands, tensor));
        break;
    case VKLLM_OP_SOFTMAX:
        _CHECK(vkllm_op_softmax_init(context, commands, tensor));
        break;
    default:
        log_error("Unknown operation type: %d", tensor->op);
        return VKLLM_ERR_ARGS;
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

    // Execute this tensor based on its operation type
    switch (tensor->op)
    {
    case VKLLM_OP_NONE:
        // Input tensors or constants, no execution needed
        break;
    case VKLLM_OP_COPY:
        _CHECK(vkllm_op_copy_run(context, commands, tensor));
        break;
    case VKLLM_OP_BIN:
        _CHECK(vkllm_op_bin_run(context, commands, tensor));
        break;
    case VKLLM_OP_EMBEDDING:
        _CHECK(vkllm_op_embedding_run(context, commands, tensor));
        break;
    case VKLLM_OP_RMSNORM:
        _CHECK(vkllm_op_rmsnorm_run(context, commands, tensor));
        break;
    case VKLLM_OP_MATMUL:
        _CHECK(vkllm_op_matmul_run(context, commands, tensor));
        break;
    case VKLLM_OP_ROPE:
        _CHECK(vkllm_op_rope_run(context, commands, tensor));
        break;
    case VKLLM_OP_SOFTMAX:
        _CHECK(vkllm_op_softmax_run(context, commands, tensor));
        break;
    default:
        log_error("Unknown operation type: %d", tensor->op);
        return VKLLM_ERR_ARGS;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_graph_run(struct vkllm_context *context, struct vkllm_graph *graph)
{
    _CHECK_ARGS(context && graph && graph->output_node && graph->commands);

    // Create a hash set to track visited tensors for O(1) lookup
    struct vkllm_hashset *visited = NULL;
    vkllm_err_t err = vkllm_hashset_new(&visited, graph->nodes->used_n);
    if (err != VKLLM_ERR_OK)
    {
        return err;
    }

    // Recursively execute starting from the output node
    err = vkllm_graph_run_tensor(context, graph->commands, graph->output_node, visited);

    // Clean up the visited hash set
    vkllm_hashset_free(visited);

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

    // Post-run this tensor based on its operation type
    switch (tensor->op)
    {
    case VKLLM_OP_NONE:
        // Input tensors or constants, no post-run needed
        break;
    case VKLLM_OP_COPY:
        _CHECK(vkllm_op_copy_post_run(context, commands, tensor));
        break;
    case VKLLM_OP_BIN:
        _CHECK(vkllm_op_bin_post_run(context, commands, tensor));
        break;
    case VKLLM_OP_EMBEDDING:
        _CHECK(vkllm_op_embedding_post_run(context, commands, tensor));
        break;
    case VKLLM_OP_RMSNORM:
        _CHECK(vkllm_op_rmsnorm_post_run(context, commands, tensor));
        break;
    case VKLLM_OP_MATMUL:
        _CHECK(vkllm_op_matmul_post_run(context, commands, tensor));
        break;
    case VKLLM_OP_ROPE:
        _CHECK(vkllm_op_rope_post_run(context, commands, tensor));
        break;
    case VKLLM_OP_SOFTMAX:
        _CHECK(vkllm_op_softmax_post_run(context, commands, tensor));
        break;
    default:
        log_error("Unknown operation type: %d", tensor->op);
        return VKLLM_ERR_ARGS;
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

    // Clean up the visited hash set
    vkllm_hashset_free(visited);

    return err;
}
