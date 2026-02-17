#include "vkllm_dtypes.h"

const char *vkllm_dtype_s(vkllm_dtype_t dtype) {
#define _VKLLM_DTYPE_OP(_dtype) #_dtype,
  static const char *vkllm_dtypes_table[] = {
#include "vkllm_dtypes.inc"
  };
#undef _VKLLM_DTYPE_OP

  return vkllm_dtypes_table[dtype - 1];
}
