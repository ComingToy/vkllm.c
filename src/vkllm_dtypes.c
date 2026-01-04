#include "vkllm_dtypes.h"

const char *vkllm_dtype_s(vkllm_dtype_t dtype)
{
#define _VKLLM_DTYPE_OP(_dtype) #_dtype,
    static const char *vkllm_dtypes_table[] = {
        "vkllm_dtype_start",
#include "vkllm_dtypes.inc"
    };
#undef _VKLLM_DTYPE_OP

    return vkllm_dtypes_table[dtype];
}

vkllm_err_t vkllm_get_dtype_info(vkllm_dtype_t dtype, struct vkllm_dtype_info *info)
{
    info->dtype = dtype;
    if (dtype == vkllm_dtype_float32)
    {
        info->bytes = sizeof(float);
        info->items_per_block = 1;
        info->bytes_per_block = info->bytes * info->items_per_block;
    }
    else if (dtype == vkllm_dtype_float16)
    {
        info->bytes = sizeof(uint16_t);
        info->items_per_block = 1;
        info->bytes_per_block = info->bytes * info->items_per_block;
    }
    else if (dtype == vkllm_dtype_int8)
    {
        info->bytes = sizeof(int8_t);
        info->items_per_block = 1;
        info->bytes_per_block = info->bytes * info->items_per_block;
    }
    else if (dtype == vkllm_dtype_uint32)
    {
        info->bytes = sizeof(uint32_t);
        info->items_per_block = 1;
        info->bytes_per_block = info->bytes * info->items_per_block;
    }
    else
    {
        log_error("%s dtype is unsported", vkllm_dtype_s(dtype));
        return VKLLM_ERR_ARGS;
    }

    return VKLLM_ERR_OK;
}

