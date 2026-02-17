#ifndef __VKLLM_TENSOR_H__
#define __VKLLM_TENSOR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "vk_mem_alloc.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_ops.h"
#include <stdbool.h>

struct vkllm_context;
struct vkllm_pipeline;
struct vkllm_gpu_device;
struct vkllm_tensor
{
    const char *name;
    vkllm_dtype_t dtype;
    uint32_t shapes[4];
    uint32_t strides[4];
    uint32_t bytes;

    struct vkllm_gpu_device *device;
    struct
    {
        VmaAllocation allocation;
        VmaAllocationInfo alloc_info;
        VkBuffer vk_buf;
        void *host;
        bool mapped; // host_buf is mapped
    } data;

    VkAccessFlagBits access_flags;
    VkPipelineStageFlagBits pipeline_stage;

    vkllm_op_t op;
    struct vkllm_pipeline *pipeline;
    struct vkllm_tensor *srcs[VKLLM_MAX_SRCS];
    uint8_t params[];
};

extern vkllm_err_t vkllm_tensor_new(struct vkllm_context *context, const char *name, const uint32_t *shapes,
                                    vkllm_dtype_t dtype, vkllm_op_t op, struct vkllm_tensor **srcs,
                                    const uint32_t n_srcs, const uint8_t *params, size_t params_bytes, bool mapped,
                                    struct vkllm_tensor **p);
extern void vkllm_tensor_free(struct vkllm_context *context, struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_tensor_invalid_cache(struct vkllm_context *context, struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_tensor_flush_cache(struct vkllm_context *context, struct vkllm_tensor *tensor);
extern vkllm_err_t vkllm_tensor_new_staging(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                            struct vkllm_tensor **staging);
extern vkllm_err_t vkllm_tensor_reshape(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                        const uint32_t *shapes);
extern vkllm_err_t vkllm_tensor_permute(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                        const uint32_t *axis);
#ifdef __cplusplus
}
#endif
#endif
