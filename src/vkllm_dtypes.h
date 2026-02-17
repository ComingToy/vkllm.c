#ifndef __VKLLM_DTYPES_H__
#define __VKLLM_DTYPES_H__
#include <stdint.h>
typedef enum {
  VKLLM_DTYPE_START = 0,
#define _VKLLM_DTYPE_OP(_dtype) vkllm_##_dtype,
#include "vkllm_dtypes.inc"
#undef _VKLLM_DTYPE_OP
} vkllm_dtype_t;

extern const char *vkllm_dtype_s(vkllm_dtype_t dtype);
inline uint32_t vkllm_dtype_size(vkllm_dtype_t dtype) {
  if (dtype == vkllm_float32) {
    return sizeof(float);
  } else if (dtype == vkllm_float16) {
    return sizeof(uint16_t);
  } else if (dtype == vkllm_int8) {
    return sizeof(int8_t);
  } else {
    return 0;
  }
}
#endif
