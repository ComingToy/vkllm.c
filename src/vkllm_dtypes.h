#ifndef __VKLLM_DTYPES_H__
#define __VKLLM_DTYPES_H__
#include <stdint.h>
typedef enum
{
    VKLLM_DTYPE_START = 0,
#define _VKLLM_DTYPE_OP(_dtype) vkllm_##_dtype,
#include "vkllm_dtypes.inc"
#undef _VKLLM_DTYPE_OP
} vkllm_dtype_t;

struct vkllm_dtype_info_t
{
    vkllm_dtype_t dtype;
    uint32_t bytes;
    uint32_t items_per_block;
    uint32_t bytes_per_block;
};

extern const char *vkllm_dtype_s(vkllm_dtype_t dtype);
inline struct vkllm_dtype_info_t vkllm_dtype_bytes(vkllm_dtype_t dtype)
{
    struct vkllm_dtype_info_t info;
    info.dtype = dtype;
    if (dtype == vkllm_float32)
    {
        info.bytes = sizeof(float);
        info.items_per_block = 1;
        info.bytes_per_block = info.bytes * info.items_per_block;
    }
    else if (dtype == vkllm_float16)
    {
        info.bytes = sizeof(uint16_t);
        info.items_per_block = 1;
        info.bytes_per_block = info.bytes * info.items_per_block;
    }
    else if (dtype == vkllm_int8)
    {
        info.bytes = sizeof(int8_t);
        info.items_per_block = 1;
        info.bytes_per_block = info.bytes * info.items_per_block;
    }else
       {
       }

    return info;
}
#endif
