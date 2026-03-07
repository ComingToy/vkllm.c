#ifndef __VKLLM_HASHMAP_H__
#define __VKLLM_HASHMAP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_common.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct vkllm_hashmap_entry
{
    char *key;
    uint64_t value;
    bool occupied;
};

struct vkllm_hashmap
{
    size_t capacity;
    size_t size;
    size_t tombstones;
    struct vkllm_hashmap_entry *entries;
};

extern uint64_t vkllm_hashmap_hash(const char *key, size_t capacity);

extern vkllm_err_t vkllm_hashmap_new(struct vkllm_hashmap **map, size_t init_capacity);

extern void vkllm_hashmap_free(struct vkllm_hashmap *map);

extern vkllm_err_t vkllm_hashmap_resize(struct vkllm_hashmap *map, size_t new_capacity);

extern vkllm_err_t vkllm_hashmap_insert(struct vkllm_hashmap *map, const char *key, uint64_t value);

extern bool vkllm_hashmap_get(const struct vkllm_hashmap *map, const char *key, uint64_t *out_value);

extern bool vkllm_hashmap_contains(const struct vkllm_hashmap *map, const char *key);

extern vkllm_err_t vkllm_hashmap_remove(struct vkllm_hashmap *map, const char *key);

extern void vkllm_hashmap_clear(struct vkllm_hashmap *map);

extern size_t vkllm_hashmap_size(const struct vkllm_hashmap *map);

extern bool vkllm_hashmap_empty(const struct vkllm_hashmap *map);

#ifdef __cplusplus
}
#endif
#endif
