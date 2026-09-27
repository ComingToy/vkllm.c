#ifndef __VKLLM_GRAPH_H__
#define __VKLLM_GRAPH_H__

#include "vkllm_array.h"

struct vkllm_tensor;
struct vkllm_context;

VKLLM_DEF_ARRAY(tensor, struct vkllm_tensor *);

struct vkllm_graph
{
    struct vkllm_array_tensor *nodes;
    struct vkllm_array_tensor *input_nodes;
    // Tensors the graph executes but does not own (e.g. model-owned KV
    // caches). Traversed like nodes; not freed by vkllm_graph_free.
    struct vkllm_array_tensor *external_nodes;
    struct vkllm_tensor *output_node;
    struct vkllm_commands *commands;
};

extern vkllm_err_t vkllm_graph_new(struct vkllm_context *context, struct vkllm_graph **graph);
extern vkllm_err_t vkllm_graph_add_input(struct vkllm_context *context, struct vkllm_graph *graph,
                                         struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_graph_set_output(struct vkllm_context *context, struct vkllm_graph *graph,
                                          struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_graph_add_node(struct vkllm_context *context, struct vkllm_graph *graph,
                                        struct vkllm_tensor *tensor);
// Linear name lookup over graph->nodes. Returns NULL if not found.
extern struct vkllm_tensor *vkllm_graph_find_node(struct vkllm_graph *graph, const char *name);
// Register a tensor the graph must execute but does NOT own (model-owned KV
// caches etc.). Traversed like nodes; not freed by vkllm_graph_free.
extern vkllm_err_t vkllm_graph_add_external(struct vkllm_context *context, struct vkllm_graph *graph,
                                            struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_graph_init(struct vkllm_context *context, struct vkllm_graph *graph);
extern vkllm_err_t vkllm_graph_run(struct vkllm_context *context, struct vkllm_graph *graph);
extern vkllm_err_t vkllm_graph_post_run(struct vkllm_context *context, struct vkllm_graph *graph);
extern vkllm_err_t vkllm_graph_free(struct vkllm_context *context, struct vkllm_graph *graph);

#endif
