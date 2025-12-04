#ifndef __VKLLM_TENSOR_H__
#define __VKLLM_TENSOR_H__
#include <stdint.h>

#include "vk_mem_alloc.h"
#include "vkllm_common.h"
#include "vkllm_context.h"
#include "vkllm_dtypes.h"
#include "vkllm_ops.h"
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

  vkllm_op_t op;
  struct vkllm_tensor *srcs[VKLLM_MAX_SRCS];
  void *params;
  bool mapped; // host_buf is mapped
};

extern struct vkllm_tensor *
vkllm_new_tensor(struct vkllm_context *context, const uint32_t *shapes,
                 const uint32_t n_shape, struct vkllm_tensor *srcs,
                 const uint32_t n_srcs, void *params, bool mapped);
#endif
