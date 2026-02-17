#ifndef __VKLLM_TENSOR_H__
#define __VKLLM_TENSOR_H__
#include <stdint.h>

typedef enum
{
    VKLLM_DTYPE_START = 0,
#define _VKLLM_DTYPE_OP(_dtype) vkllm_##_dtype,
#include "vkllm_dtypes.inc"
#undef _VKLLM_DTYPE_OP
} vkllm_dtype_t;

struct vkllm_tensor
{
	char* name;
	void* buf;
	vkllm_dtype_t dtype;
	uint32_t shapes[4];
	uint32_t strides[4];
};

#endif
