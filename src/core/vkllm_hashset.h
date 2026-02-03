#ifndef __VKLLM_HASHSET_H__
#define __VKLLM_HASHSET_H__

#ifdef __cplusplus
extern "C"
{
#endif

#include "vkllm_common.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Hash set bucket entry
struct vkllm_hashset_entry
{
    uint64_t key;
    bool occupied;
};

// Hash set structure with open addressing and linear probing
struct vkllm_hashset
{
    size_t capacity;     // Total number of buckets
    size_t size;         // Number of occupied entries
    size_t tombstones;   // Number of deleted entries
    struct vkllm_hashset_entry *entries;
};

// Hash function for uint64_t keys
extern uint64_t vkllm_hashset_hash(uint64_t key, size_t capacity);

// Create a new hash set with initial capacity
extern vkllm_err_t vkllm_hashset_new(struct vkllm_hashset **set, size_t init_capacity);

// Free a hash set
extern void vkllm_hashset_free(struct vkllm_hashset *set);

// Resize the hash set (internal helper)
extern vkllm_err_t vkllm_hashset_resize(struct vkllm_hashset *set, size_t new_capacity);

// Insert a key into the hash set
// Returns VKLLM_ERR_OK if inserted successfully or key already exists
extern vkllm_err_t vkllm_hashset_insert(struct vkllm_hashset *set, uint64_t key);

// Check if a key exists in the hash set
extern bool vkllm_hashset_contains(const struct vkllm_hashset *set, uint64_t key);

// Remove a key from the hash set
extern vkllm_err_t vkllm_hashset_remove(struct vkllm_hashset *set, uint64_t key);

// Clear all entries from the hash set
extern void vkllm_hashset_clear(struct vkllm_hashset *set);

// Get the number of elements in the hash set
extern size_t vkllm_hashset_size(const struct vkllm_hashset *set);

// Check if the hash set is empty
extern bool vkllm_hashset_empty(const struct vkllm_hashset *set);

#ifdef __cplusplus
}
#endif
#endif
