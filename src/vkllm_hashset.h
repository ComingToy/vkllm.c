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
#include <stdlib.h>
#include <string.h>

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

// FNV-1a hash function for uint64_t
static inline uint64_t vkllm_hashset_hash(uint64_t key, size_t capacity)
{
    // Simple modulo hash with multiplicative mixing
    // Using a prime multiplier for better distribution
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key % capacity;
}

// Create a new hash set with initial capacity
static inline vkllm_err_t vkllm_hashset_new(struct vkllm_hashset **set, size_t init_capacity)
{
    if (!set || init_capacity == 0)
    {
        return VKLLM_ERR_ARGS;
    }

    *set = (struct vkllm_hashset *)malloc(sizeof(struct vkllm_hashset));
    if (*set == NULL)
    {
        return VKLLM_ERR_ALLOC;
    }

    // Ensure capacity is at least 16 for good performance
    if (init_capacity < 16)
    {
        init_capacity = 16;
    }

    (*set)->capacity = init_capacity;
    (*set)->size = 0;
    (*set)->tombstones = 0;
    (*set)->entries = (struct vkllm_hashset_entry *)calloc(init_capacity, sizeof(struct vkllm_hashset_entry));
    
    if ((*set)->entries == NULL)
    {
        free(*set);
        *set = NULL;
        return VKLLM_ERR_ALLOC;
    }

    return VKLLM_ERR_OK;
}

// Free a hash set
static inline void vkllm_hashset_free(struct vkllm_hashset *set)
{
    if (!set)
        return;
    if (set->entries)
        free(set->entries);
    free(set);
}

// Resize the hash set (internal helper)
static inline vkllm_err_t vkllm_hashset_resize(struct vkllm_hashset *set, size_t new_capacity)
{
    if (!set)
    {
        return VKLLM_ERR_ARGS;
    }

    struct vkllm_hashset_entry *old_entries = set->entries;
    size_t old_capacity = set->capacity;

    set->entries = (struct vkllm_hashset_entry *)calloc(new_capacity, sizeof(struct vkllm_hashset_entry));
    if (set->entries == NULL)
    {
        set->entries = old_entries;
        return VKLLM_ERR_ALLOC;
    }

    set->capacity = new_capacity;
    set->size = 0;
    set->tombstones = 0;

    // Rehash all existing entries
    for (size_t i = 0; i < old_capacity; i++)
    {
        if (old_entries[i].occupied)
        {
            uint64_t hash = vkllm_hashset_hash(old_entries[i].key, new_capacity);
            
            // Linear probing to find an empty slot
            for (size_t j = 0; j < new_capacity; j++)
            {
                size_t idx = (hash + j) % new_capacity;
                if (!set->entries[idx].occupied)
                {
                    set->entries[idx].key = old_entries[i].key;
                    set->entries[idx].occupied = true;
                    set->size++;
                    break;
                }
            }
        }
    }

    free(old_entries);
    return VKLLM_ERR_OK;
}

// Insert a key into the hash set
// Returns VKLLM_ERR_OK if inserted successfully or key already exists
static inline vkllm_err_t vkllm_hashset_insert(struct vkllm_hashset *set, uint64_t key)
{
    if (!set)
    {
        return VKLLM_ERR_ARGS;
    }

    // Resize if load factor exceeds 0.75
    if ((set->size + set->tombstones) * 4 >= set->capacity * 3)
    {
        size_t new_capacity = set->capacity * 2;
        vkllm_err_t err = vkllm_hashset_resize(set, new_capacity);
        if (err != VKLLM_ERR_OK)
        {
            return err;
        }
    }

    uint64_t hash = vkllm_hashset_hash(key, set->capacity);

    // Linear probing to find insertion position
    for (size_t i = 0; i < set->capacity; i++)
    {
        size_t idx = (hash + i) % set->capacity;
        
        if (!set->entries[idx].occupied)
        {
            // Found empty slot, insert here
            set->entries[idx].key = key;
            set->entries[idx].occupied = true;
            set->size++;
            return VKLLM_ERR_OK;
        }
        else if (set->entries[idx].key == key)
        {
            // Key already exists
            return VKLLM_ERR_OK;
        }
    }

    // Should never reach here if resize works correctly
    return VKLLM_ERR_ALLOC;
}

// Check if a key exists in the hash set
static inline bool vkllm_hashset_contains(const struct vkllm_hashset *set, uint64_t key)
{
    if (!set || set->size == 0)
    {
        return false;
    }

    uint64_t hash = vkllm_hashset_hash(key, set->capacity);

    // Linear probing to find the key
    for (size_t i = 0; i < set->capacity; i++)
    {
        size_t idx = (hash + i) % set->capacity;
        
        if (!set->entries[idx].occupied)
        {
            // Found empty slot, key doesn't exist
            return false;
        }
        else if (set->entries[idx].key == key)
        {
            // Found the key
            return true;
        }
    }

    return false;
}

// Remove a key from the hash set
static inline vkllm_err_t vkllm_hashset_remove(struct vkllm_hashset *set, uint64_t key)
{
    if (!set)
    {
        return VKLLM_ERR_ARGS;
    }

    uint64_t hash = vkllm_hashset_hash(key, set->capacity);

    // Linear probing to find the key
    for (size_t i = 0; i < set->capacity; i++)
    {
        size_t idx = (hash + i) % set->capacity;
        
        if (!set->entries[idx].occupied)
        {
            // Key not found
            return VKLLM_ERR_OK;
        }
        else if (set->entries[idx].key == key)
        {
            // Found the key, mark as deleted
            set->entries[idx].occupied = false;
            set->entries[idx].key = 0;
            set->size--;
            set->tombstones++;
            return VKLLM_ERR_OK;
        }
    }

    return VKLLM_ERR_OK;
}

// Clear all entries from the hash set
static inline void vkllm_hashset_clear(struct vkllm_hashset *set)
{
    if (!set)
        return;
    
    memset(set->entries, 0, set->capacity * sizeof(struct vkllm_hashset_entry));
    set->size = 0;
    set->tombstones = 0;
}

// Get the number of elements in the hash set
static inline size_t vkllm_hashset_size(const struct vkllm_hashset *set)
{
    return set ? set->size : 0;
}

// Check if the hash set is empty
static inline bool vkllm_hashset_empty(const struct vkllm_hashset *set)
{
    return !set || set->size == 0;
}

#ifdef __cplusplus
}
#endif
#endif
