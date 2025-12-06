#ifndef __VKLLM_ARRAY_H__
#define __VKLLM_ARRAY_H__

#include "vkllm_common.h"
#include "vkllm_errors.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define VKLLM_DEF_ARRAY(_name, _type)                                                                                  \
    struct vkllm_array_##_name                                                                                         \
    {                                                                                                                  \
        size_t alloc_n;                                                                                                \
        size_t used_n;                                                                                                 \
        _type *data;                                                                                                   \
    };                                                                                                                 \
                                                                                                                       \
    static inline vkllm_err_t vkllm_array_##_name##_new(struct vkllm_array_##_name **arr, size_t init)                 \
    {                                                                                                                  \
        size_t alloc_bytes = sizeof(**arr);                                                                            \
        *arr = (struct vkllm_array_##_name *)malloc(alloc_bytes);                                                      \
        if (*arr == NULL)                                                                                              \
        {                                                                                                              \
            return VKLLM_ERR_ALLOC;                                                                                    \
        }                                                                                                              \
        (*arr)->data = NULL;                                                                                           \
        (*arr)->alloc_n = init;                                                                                        \
        (*arr)->used_n = 0;                                                                                            \
        if (init > 0)                                                                                                  \
        {                                                                                                              \
            _NEW_N_AND_CHECK((*arr)->data, _type, init);                                                               \
        }                                                                                                              \
        return VKLLM_ERR_OK;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    static inline vkllm_err_t vkllm_array_##_name##_append(struct vkllm_array_##_name *arr, _type element)             \
    {                                                                                                                  \
        if (arr->used_n >= arr->alloc_n)                                                                               \
        {                                                                                                              \
            _type *data = NULL;                                                                                        \
            _NEW_N_AND_CHECK(data, _type, arr->alloc_n * 2);                                                           \
            memcpy(data, arr->data, arr->alloc_n * sizeof(_type));                                                     \
            free(arr->data);                                                                                           \
            arr->data = data;                                                                                          \
            arr->alloc_n *= 2;                                                                                         \
        }                                                                                                              \
        arr->data[arr->used_n++] = element;                                                                            \
        return VKLLM_ERR_OK;                                                                                           \
    }                                                                                                                  \
    static inline void vkllm_array_##_name##_free(struct vkllm_array_##_name *arr)                                     \
    {                                                                                                                  \
        if (!arr)                                                                                                      \
            return;                                                                                                    \
        if (arr->data)                                                                                                 \
            free(arr->data);                                                                                           \
        free(arr);                                                                                                     \
    }

VKLLM_DEF_ARRAY(ptr, void *)
#endif
