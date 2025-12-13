#ifndef __VKLLM_DTYPES_H__
#define __VKLLM_DTYPES_H__
#include "vkllm_errors.h"
#include <log.h>
#include <stdint.h>
typedef enum
{
    vkllm_dtype_start = 0,
#define _VKLLM_DTYPE_OP(_dtype) vkllm_dtype_##_dtype,
#include "vkllm_dtypes.inc"
#undef _VKLLM_DTYPE_OP
} vkllm_dtype_t;

struct vkllm_dtype_info
{
    vkllm_dtype_t dtype;
    uint32_t bytes;
    uint32_t items_per_block;
    uint32_t bytes_per_block;
};

extern const char *vkllm_dtype_s(vkllm_dtype_t dtype);
extern vkllm_err_t vkllm_get_dtype_info(vkllm_dtype_t dtype, struct vkllm_dtype_info *info);
#endif
