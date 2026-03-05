#ifndef __VKLLM_KVCACHE_H__
#define __VKLLM_KVCACHE_H__

#include "src/core/vkllm_errors.h"
#include <stdint.h>

struct vkllm_array_tensor;
struct vkllm_tensor;
struct vkllm_context;
struct vkllm_graph;
struct vkllm_kvcache
{
    struct vkllm_array_tensor *kcaches;
    struct vkllm_array_tensor *vcaches;
    uint32_t cache_shape[4];
};

vkllm_err_t vkllm_kvcache_new(struct vkllm_context *context, uint32_t cache_shape[4], uint32_t layer_counts,
                              struct vkllm_kvcache **kvcache);
vkllm_err_t vkllm_kvcache_update(struct vkllm_context *context, struct vkllm_kvcache *kvcache,
                                 struct vkllm_tensor **key, struct vkllm_tensor **value, uint32_t layer,
                                 uint32_t offset);
void vkllm_kvcache_free(struct vkllm_context *context, struct vkllm_kvcache *kvcache);
#endif
