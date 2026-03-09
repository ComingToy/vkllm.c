#include "vkllm_hashmap.h"
#include <stdlib.h>
#include <string.h>

static char *vkllm_hashmap_strdup(const char *s)
{
    size_t len = strlen(s) + 1;
    char *dup = (char *)malloc(len);
    if (dup)
        memcpy(dup, s, len);
    return dup;
}

uint64_t vkllm_hashmap_hash(const char *key, size_t capacity)
{
    uint64_t hash = 14695981039346656037ULL;
    while (*key)
    {
        hash ^= (uint8_t)(*key);
        hash *= 1099511628211ULL;
        key++;
    }
    return hash % capacity;
}

vkllm_err_t vkllm_hashmap_new(struct vkllm_hashmap **map, size_t init_capacity)
{
    if (!map || init_capacity == 0)
        return VKLLM_ERR_ARGS;

    *map = (struct vkllm_hashmap *)malloc(sizeof(struct vkllm_hashmap));
    if (*map == NULL)
        return VKLLM_ERR_ALLOC;

    if (init_capacity < 16)
        init_capacity = 16;

    (*map)->capacity = init_capacity;
    (*map)->size = 0;
    (*map)->tombstones = 0;
    (*map)->entries = (struct vkllm_hashmap_entry *)calloc(init_capacity, sizeof(struct vkllm_hashmap_entry));

    if ((*map)->entries == NULL)
    {
        free(*map);
        *map = NULL;
        return VKLLM_ERR_ALLOC;
    }

    return VKLLM_ERR_OK;
}

void vkllm_hashmap_free(struct vkllm_hashmap *map)
{
    if (!map)
        return;
    if (map->entries)
    {
        for (size_t i = 0; i < map->capacity; i++)
        {
            if (map->entries[i].key)
                free(map->entries[i].key);
        }
        free(map->entries);
    }
    free(map);
}

vkllm_err_t vkllm_hashmap_resize(struct vkllm_hashmap *map, size_t new_capacity)
{
    if (!map)
        return VKLLM_ERR_ARGS;

    struct vkllm_hashmap_entry *old_entries = map->entries;
    size_t old_capacity = map->capacity;

    map->entries = (struct vkllm_hashmap_entry *)calloc(new_capacity, sizeof(struct vkllm_hashmap_entry));
    if (map->entries == NULL)
    {
        map->entries = old_entries;
        return VKLLM_ERR_ALLOC;
    }

    map->capacity = new_capacity;
    map->size = 0;
    map->tombstones = 0;

    for (size_t i = 0; i < old_capacity; i++)
    {
        if (old_entries[i].occupied && old_entries[i].key)
        {
            uint64_t hash = vkllm_hashmap_hash(old_entries[i].key, new_capacity);

            for (size_t j = 0; j < new_capacity; j++)
            {
                size_t idx = (hash + j) % new_capacity;
                if (!map->entries[idx].occupied)
                {
                    map->entries[idx].key = old_entries[i].key;
                    map->entries[idx].value = old_entries[i].value;
                    map->entries[idx].occupied = true;
                    map->size++;
                    break;
                }
            }
        }
        else if (old_entries[i].key)
        {
            free(old_entries[i].key);
        }
    }

    free(old_entries);
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_hashmap_insert(struct vkllm_hashmap *map, const char *key, uint64_t value)
{
    if (!map || !key)
        return VKLLM_ERR_ARGS;

    if ((map->size + map->tombstones) * 4 >= map->capacity * 3)
    {
        size_t new_capacity = map->capacity * 2;
        vkllm_err_t err = vkllm_hashmap_resize(map, new_capacity);
        if (err != VKLLM_ERR_OK)
            return err;
    }

    uint64_t hash = vkllm_hashmap_hash(key, map->capacity);

    for (size_t i = 0; i < map->capacity; i++)
    {
        size_t idx = (hash + i) % map->capacity;

        if (!map->entries[idx].occupied)
        {
            char *key_copy = vkllm_hashmap_strdup(key);
            if (!key_copy)
                return VKLLM_ERR_ALLOC;

            if (map->entries[idx].key)
                free(map->entries[idx].key);

            map->entries[idx].key = key_copy;
            map->entries[idx].value = value;
            map->entries[idx].occupied = true;
            map->size++;
            return VKLLM_ERR_OK;
        }
        else if (map->entries[idx].key && strcmp(map->entries[idx].key, key) == 0)
        {
            map->entries[idx].value = value;
            return VKLLM_ERR_OK;
        }
    }

    return VKLLM_ERR_ALLOC;
}

bool vkllm_hashmap_get(const struct vkllm_hashmap *map, const char *key, uint64_t *out_value)
{
    if (!map || !key || map->size == 0)
        return false;

    uint64_t hash = vkllm_hashmap_hash(key, map->capacity);

    for (size_t i = 0; i < map->capacity; i++)
    {
        size_t idx = (hash + i) % map->capacity;

        if (!map->entries[idx].occupied && !map->entries[idx].key)
            return false;
        else if (map->entries[idx].key && strcmp(map->entries[idx].key, key) == 0)
        {
            if (out_value)
                *out_value = map->entries[idx].value;
            return true;
        }
    }

    return false;
}

bool vkllm_hashmap_contains(const struct vkllm_hashmap *map, const char *key)
{
    return vkllm_hashmap_get(map, key, NULL);
}

vkllm_err_t vkllm_hashmap_remove(struct vkllm_hashmap *map, const char *key)
{
    if (!map || !key)
        return VKLLM_ERR_ARGS;

    uint64_t hash = vkllm_hashmap_hash(key, map->capacity);

    for (size_t i = 0; i < map->capacity; i++)
    {
        size_t idx = (hash + i) % map->capacity;

        if (!map->entries[idx].occupied && !map->entries[idx].key)
            return VKLLM_ERR_OK;
        else if (map->entries[idx].key && strcmp(map->entries[idx].key, key) == 0)
        {
            free(map->entries[idx].key);
            map->entries[idx].key = NULL;
            map->entries[idx].value = 0;
            map->entries[idx].occupied = false;
            map->size--;
            map->tombstones++;
            return VKLLM_ERR_OK;
        }
    }

    return VKLLM_ERR_OK;
}

void vkllm_hashmap_clear(struct vkllm_hashmap *map)
{
    if (!map)
        return;

    for (size_t i = 0; i < map->capacity; i++)
    {
        if (map->entries[i].key)
        {
            free(map->entries[i].key);
            map->entries[i].key = NULL;
        }
        map->entries[i].value = 0;
        map->entries[i].occupied = false;
    }
    map->size = 0;
    map->tombstones = 0;
}

size_t vkllm_hashmap_size(const struct vkllm_hashmap *map)
{
    return map ? map->size : 0;
}

bool vkllm_hashmap_empty(const struct vkllm_hashmap *map)
{
    return !map || map->size == 0;
}
