#ifndef __VKLLM_KVCACHE_H__
#define __VKLLM_KVCACHE_H__

#include "src/core/vkllm_errors.h"
#include <stdint.h>

struct vkllm_array_tensor;
struct vkllm_tensor;
struct vkllm_context;
struct vkllm_graph;

// Per-layer K/V cache.
//
// The caches are pre-allocated at full context length. During graph build,
// `vkllm_kvcache_bind` hands out persistent slice views (`kview`/`vview`)
// covering [0, cur_len) rows; the returned views are registered into the
// graph once. Each subsequent decode step only calls `vkllm_kvcache_advance`,
// which updates the update-rows offset and shrinks the views in place —
// no allocations, no descriptor updates.
struct vkllm_kvcache
{
    struct vkllm_array_tensor *kcaches;
    struct vkllm_array_tensor *vcaches;
    // Persistent slice views into the caches, one per layer, parallel to
    // kcaches/vcaches. NULL until vkllm_kvcache_bind is called.
    struct vkllm_array_tensor *kviews;
    struct vkllm_array_tensor *vviews;
    uint32_t kcache_shape[4];
    uint32_t vcache_shape[4];
    uint32_t layer_counts;
    // Current logical length (rows of K/V written so far, including this
    // step's input once bound).
    uint32_t cur_len;
};

vkllm_err_t vkllm_kvcache_new(struct vkllm_context *context, uint32_t kcache_shape[4], uint32_t vcache_shape[4],
                              uint32_t layer_counts, struct vkllm_kvcache **kvcache);

// Bind layer `layer`'s key/value producers into its cache and create the
// persistent views consumed by attention. On return, `*key` / `*value` are
// replaced with the view tensors (ownership stays with the kvcache — do NOT
// add them to graph->nodes via add_node's consuming path twice; register
// them once per graph build).
vkllm_err_t vkllm_kvcache_bind(struct vkllm_context *context, struct vkllm_kvcache *kvcache,
                               struct vkllm_graph *graph, struct vkllm_tensor **key, struct vkllm_tensor **value,
                               uint32_t layer, uint32_t offset);

// Advance to the next decode step: the views grow to `offset + seq_len` rows
// and the update-rows offset moves to `offset`. Cheap — no allocations.
vkllm_err_t vkllm_kvcache_advance(struct vkllm_context *context, struct vkllm_kvcache *kvcache, uint32_t offset,
                                  uint32_t seq_len);

void vkllm_kvcache_free(struct vkllm_context *context, struct vkllm_kvcache *kvcache);
#endif
