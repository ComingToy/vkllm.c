#ifndef __VKLLM_TENSOR_H__
#define __VKLLM_TENSOR_H__
#include <stdint.h>

#include "vk_mem_alloc.h"
#include "vkllm_dtypes.h"
#include "vkllm_common.h"
#include <stdbool.h>

struct vkllm_tensor {
  char *name;
  void *host_buf;
  vkllm_dtype_t dtype;
  uint32_t shapes[4];
  uint32_t strides[4];
  uint32_t bytes;
  struct {
    VmaAllocation allocation;
    VmaAllocationInfo alloc_info;
    VkBuffer buf;
  } device_buf;

  struct vkllm_tensor* srcs[VKLLM_MAX_SRCS];
  void* params;
  bool visable; // host_buf is mapped
};

#endif
