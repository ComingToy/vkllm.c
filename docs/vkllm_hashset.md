# vkllm_hashset - Hash Set Container

A high-performance hash set implementation for uint64_t keys with O(1) average-case lookup, insertion, and deletion.

## Features

- **Fast O(1) operations**: Average-case constant time for lookup, insertion, and deletion
- **Open addressing with linear probing**: Cache-friendly implementation
- **Automatic resizing**: Maintains load factor ≤ 0.75 for optimal performance
- **Header-only**: All functions are inline for maximum performance
- **Memory efficient**: Uses calloc for zero-initialization and minimal memory overhead

## API Reference

### Data Structures

```c
struct vkllm_hashset
{
    size_t capacity;     // Total number of buckets
    size_t size;         // Number of occupied entries
    size_t tombstones;   // Number of deleted entries
    struct vkllm_hashset_entry *entries;
};
```

### Core Functions

#### Create and Destroy

```c
vkllm_err_t vkllm_hashset_new(struct vkllm_hashset **set, size_t init_capacity);
void vkllm_hashset_free(struct vkllm_hashset *set);
```

Create a new hash set with specified initial capacity (minimum 16).
Free all resources associated with the hash set.

#### Insert and Check

```c
vkllm_err_t vkllm_hashset_insert(struct vkllm_hashset *set, uint64_t key);
bool vkllm_hashset_contains(const struct vkllm_hashset *set, uint64_t key);
```

Insert a key into the set (no-op if already exists).
Check if a key exists in the set (O(1) average case).

#### Remove and Clear

```c
vkllm_err_t vkllm_hashset_remove(struct vkllm_hashset *set, uint64_t key);
void vkllm_hashset_clear(struct vkllm_hashset *set);
```

Remove a key from the set (no-op if doesn't exist).
Remove all entries from the set.

#### Utility Functions

```c
size_t vkllm_hashset_size(const struct vkllm_hashset *set);
bool vkllm_hashset_empty(const struct vkllm_hashset *set);
```

Get the number of elements in the set.
Check if the set is empty.

## Usage Example

```c
#include "vkllm_hashset.h"

// Create a hash set
struct vkllm_hashset *set = NULL;
vkllm_err_t err = vkllm_hashset_new(&set, 16);
if (err != VKLLM_ERR_OK) {
    // Handle error
    return err;
}

// Insert keys
vkllm_hashset_insert(set, 42);
vkllm_hashset_insert(set, 123);
vkllm_hashset_insert(set, 999);

// Check if keys exist
if (vkllm_hashset_contains(set, 42)) {
    printf("Key 42 exists\n");
}

// Get size
printf("Set size: %zu\n", vkllm_hashset_size(set));

// Remove a key
vkllm_hashset_remove(set, 123);

// Clear all entries
vkllm_hashset_clear(set);

// Free the set
vkllm_hashset_free(set);
```

## Use Cases

### Graph Traversal (vkllm_graph.c)

The hash set is used in `vkllm_graph_init` to track visited tensors during graph initialization:

```c
// Create a hash set to track visited tensors for O(1) lookup
struct vkllm_hashset *visited = NULL;
vkllm_err_t err = vkllm_hashset_new(&visited, graph->nodes->used_n);

// Check if tensor already visited
uint64_t tensor_key = (uint64_t)(uintptr_t)tensor;
if (vkllm_hashset_contains(visited, tensor_key)) {
    return VKLLM_ERR_OK; // Already visited
}

// Mark as visited
vkllm_hashset_insert(visited, tensor_key);

// Clean up
vkllm_hashset_free(visited);
```

### Pointer Deduplication

Track unique pointers by casting to uint64_t:

```c
struct vkllm_hashset *unique_ptrs = NULL;
vkllm_hashset_new(&unique_ptrs, 32);

for (size_t i = 0; i < n; i++) {
    uint64_t ptr_key = (uint64_t)(uintptr_t)ptrs[i];
    if (!vkllm_hashset_contains(unique_ptrs, ptr_key)) {
        vkllm_hashset_insert(unique_ptrs, ptr_key);
        // Process unique pointer
        process(ptrs[i]);
    }
}

vkllm_hashset_free(unique_ptrs);
```

## Implementation Details

### Hash Function

Uses a high-quality multiplicative hash function based on MurmurHash3's finalizer:

```c
static inline uint64_t vkllm_hashset_hash(uint64_t key, size_t capacity)
{
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key % capacity;
}
```

### Collision Resolution

Uses linear probing with the following strategy:
- Primary hash: `hash(key) % capacity`
- Probe sequence: `(hash + i) % capacity` for i = 0, 1, 2, ...

### Load Factor Management

Automatically resizes when load factor exceeds 0.75:
- Doubles capacity on resize
- Rehashes all existing entries
- Maintains optimal performance characteristics

## Performance

- **Lookup**: O(1) average, O(n) worst case
- **Insertion**: O(1) average, O(n) worst case
- **Deletion**: O(1) average, O(n) worst case
- **Space**: O(n) where n is the number of inserted elements

## Testing

Run the comprehensive test suite:

```bash
bazel test //tests:vkllm_test_hashset
```

Tests cover:
- Basic insertion and lookup
- Duplicate key handling
- Removal operations
- Automatic resizing
- Large keys (64-bit values)
- Collision handling
- Clear operations
