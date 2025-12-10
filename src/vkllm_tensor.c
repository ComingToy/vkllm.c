#include "vkllm_tensor.h"
#include "vkllm_dtypes.h"

#include "src/vkllm_common.h"
#include <string.h>

static vkllm_err_t vkllm_calc_strides(const uint32_t *shapes, vkllm_dtype_t dtype, uint32_t *strides)
{
    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(dtype, &dtype_info));

    uint32_t w = shapes[3];
    uint32_t blocks = (w + dtype_info.items_per_block - 1) / dtype_info.items_per_block;
    uint32_t bytes = blocks * dtype_info.bytes_per_block;

    strides[3] = dtype_info.bytes;
    strides[2] = bytes;
    strides[1] = shapes[1] * strides[2];
    strides[0] = shapes[0] * strides[1];

    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_create_vk_buffer(struct vkllm_tensor *tensor)
{
    VkBufferCreateInfo buffer_create_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                             .pNext = NULL,
                                             .flags = 0,
                                             .size = tensor->bytes,
                                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                             .queueFamilyIndexCount = 0,
                                             .pQueueFamilyIndices = NULL};

    VmaAllocationCreateFlags flags = 0;
    if (tensor->data.mapped)
    {
        flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VmaAllocationCreateInfo alloc_create_info = {.flags = flags, .usage = VMA_MEMORY_USAGE_AUTO};
    VkResult err = vmaCreateBuffer(tensor->device->vma_allocator, &buffer_create_info, &alloc_create_info,
                                   &tensor->data.vk_buf, &tensor->data.allocation, &tensor->data.alloc_info);
    if (err != VK_SUCCESS)
    {
        log_error("vmaCreateBuffer failed: %d", (int)err);
        return VKLLM_ERR_VULKAN;
    }

    tensor->data.host = tensor->data.mapped ? tensor->data.alloc_info.pMappedData : NULL;
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_tensor_new(struct vkllm_context *context, struct vkllm_gpu_device *device, const char *name,
                             const uint32_t *shapes, vkllm_dtype_t dtype, vkllm_op_t op, struct vkllm_tensor **srcs,
                             const uint32_t n_srcs, const uint8_t *params, size_t params_bytes, bool mapped,
                             struct vkllm_tensor **p)
{
    if (!shapes)
    {
        log_error("shape is empty. shapes is NULL");
        return VKLLM_ERR_ARGS;
    }

    for (uint32_t i = 0; i < 4; ++i)
    {
        if (!shapes[i] || shapes[i] == 0)
        {
            log_error("shape is empty. shapes[%u] is 0", i);
            return VKLLM_ERR_ARGS;
        }
    }

    *p = (struct vkllm_tensor *)malloc(sizeof(struct vkllm_tensor) + params_bytes);

    struct vkllm_tensor *t = *p;
    t->name = name;
    t->dtype = dtype;
    t->shapes[0] = shapes[0];
    t->shapes[1] = shapes[1];
    t->shapes[2] = shapes[2];
    t->shapes[3] = shapes[3];

    vkllm_err_t err = vkllm_calc_strides(shapes, dtype, t->strides);
    if (err != VKLLM_ERR_OK)
    {
        goto err_calc_strides;
    }

    size_t total_bytes = shapes[0] * t->strides[0];
    size_t align = device->vk_physical_dev.properties.limits.nonCoherentAtomSize;
    size_t total_aligned_bytes = (total_bytes + align - 1) / align * align;
    t->bytes = total_aligned_bytes;

    t->device = device;
    t->op = op;

    for (uint32_t i = 0; i < VKLLM_MAX_SRCS; ++i)
    {
        if (i < n_srcs)
        {
            t->srcs[i] = srcs[i];
            continue;
        }
        t->srcs[i] = NULL;
    }

    if (params)
    {
        memcpy(t->params, params, params_bytes);
    }
    t->data.mapped = mapped;
    t->access_flags = 0;
    t->pipeline_stage = 0;

    err = vkllm_create_vk_buffer(t);
    if (err != VKLLM_ERR_OK)
    {
        goto err_create_vk_buf;
    }

    return VKLLM_ERR_OK;

err_calc_strides:
err_create_vk_buf:
    free(*p);
    return err;
}

void vkllm_tensor_free(struct vkllm_context *context, struct vkllm_tensor *tensor)
{
    vmaDestroyBuffer(tensor->device->vma_allocator, tensor->data.vk_buf, tensor->data.allocation);
    free(tensor);
}

vkllm_err_t vkllm_tensor_flush_cache(struct vkllm_context *context, struct vkllm_tensor *tensor)
{
    if (!tensor->data.mapped)
    {
        log_error("can only flush mapped buffer.");
        return VKLLM_ERR_ARGS;
    }

    VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                 .pNext = NULL,
                                 .memory = tensor->data.alloc_info.deviceMemory,
                                 .offset = tensor->data.alloc_info.offset,
                                 .size = tensor->data.alloc_info.size};
    _CHECK_VK(vkFlushMappedMemoryRanges(tensor->device->vk_dev, 1, &range));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_tensor_invalid_cache(struct vkllm_context *context, struct vkllm_tensor *tensor)
{
    if (!tensor->data.mapped)
    {
        log_error("can only invalid mapped buffer. ");
        return VKLLM_ERR_ARGS;
    }

    VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                 .pNext = NULL,
                                 .memory = tensor->data.alloc_info.deviceMemory,
                                 .offset = tensor->data.alloc_info.offset,
                                 .size = tensor->data.alloc_info.size};

    _CHECK_VK(vkInvalidateMappedMemoryRanges(tensor->device->vk_dev, 1, &range));
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_tensor_new_staging(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                     struct vkllm_tensor **staging)
{
    _CHECK_ARGS(context && tensor && staging);
    _CHECK(vkllm_tensor_new(context, tensor->device, "", tensor->shapes, tensor->dtype, tensor->op, tensor->srcs, 4,
                            NULL, 0, true, staging));
    return VKLLM_ERR_OK;
}
