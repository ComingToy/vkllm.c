#include "vkllm_tensor.h"
#include "vkllm_context.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_ops.h"

#include "vkllm_common.h"
#include <stdlib.h>
#include <string.h>

static vkllm_err_t vkllm_calc_strides(struct vkllm_gpu_device *device, const uint32_t *shapes, vkllm_dtype_t dtype,
                                      uint32_t *strides)
{
    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(dtype, &dtype_info));

    uint32_t w = shapes[3];
    uint32_t blocks = (w + dtype_info.items_per_block - 1) / dtype_info.items_per_block;
    uint32_t bytes = blocks * dtype_info.bytes_per_block;

    size_t align_bytes = bytes;

    strides[3] = dtype_info.bytes;
    strides[2] = (uint32_t)align_bytes;
    strides[1] = shapes[2] * strides[2];
    strides[0] = shapes[1] * strides[1];

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
    tensor->data.is_ref = false;
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_tensor_new(struct vkllm_context *context, const char *name, const uint32_t *shapes,
                             vkllm_dtype_t dtype, vkllm_op_t op, struct vkllm_tensor **srcs, const uint32_t n_srcs,
                             const void *params, size_t params_bytes, bool mapped, struct vkllm_tensor **p)
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

    struct vkllm_gpu_device *device = context->device;
    *p = (struct vkllm_tensor *)malloc(sizeof(struct vkllm_tensor) + params_bytes);

    struct vkllm_tensor *t = *p;

    strncpy((char *)t->name, name, sizeof(t->name));

    t->dtype = dtype;
    t->shapes[0] = shapes[0];
    t->shapes[1] = shapes[1];
    t->shapes[2] = shapes[2];
    t->shapes[3] = shapes[3];

    vkllm_err_t err = vkllm_calc_strides(device, shapes, dtype, t->strides);
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

    t->pipeline = NULL;
    context->stats.tensor_alloc_counts += 1;
    return VKLLM_ERR_OK;

err_calc_strides:
err_create_vk_buf:
    free(*p);
    return err;
}

void vkllm_tensor_free(struct vkllm_context *context, struct vkllm_tensor *tensor)
{
    if (!tensor->data.is_ref)
    {
        vmaDestroyBuffer(tensor->device->vma_allocator, tensor->data.vk_buf, tensor->data.allocation);
    }
    free(tensor);
    context->stats.tensor_alloc_counts -= 1;
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
    _CHECK(vkllm_tensor_new(context, "", tensor->shapes, tensor->dtype, tensor->op, tensor->srcs, 4, NULL, 0, true,
                            staging));
    return VKLLM_ERR_OK;
}

static bool is_padding(struct vkllm_context *context, vkllm_dtype_t dtype, uint32_t w)
{
    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(dtype, &dtype_info));

    uint32_t blocks = (w + dtype_info.items_per_block - 1) / dtype_info.items_per_block;
    uint32_t bytes = blocks * dtype_info.bytes_per_block;

    size_t align_bytes = bytes;
    size_t unaligned_bytes = dtype_info.bytes * w;

    return align_bytes != unaligned_bytes;
}

vkllm_err_t vkllm_tensor_reshape(struct vkllm_context *context, struct vkllm_tensor *tensor, const uint32_t *shapes)
{
    _CHECK_ARGS(context && tensor && shapes);
    _CHECK_ARGS(_MUL4(tensor->shapes) == _MUL4(shapes));
    _CHECK_ARGS(!is_padding(context, tensor->dtype, tensor->shapes[3]));
    _CHECK_ARGS(!is_padding(context, tensor->dtype, shapes[3]));

    tensor->shapes[0] = shapes[0];
    tensor->shapes[1] = shapes[1];
    tensor->shapes[2] = shapes[2];
    tensor->shapes[3] = shapes[3];

    // IMPORTANT: Recalculate strides for the new shape
    // Without this, the tensor would have incorrect strides and data access would be wrong
    _CHECK(vkllm_calc_strides(tensor->device, shapes, tensor->dtype, tensor->strides));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_tensor_permute(struct vkllm_context *context, struct vkllm_tensor *tensor, const uint32_t *axis)
{
    _CHECK_ARGS(context && tensor && axis);

    uint32_t shapes[4] = {tensor->shapes[0], tensor->shapes[1], tensor->shapes[2], tensor->shapes[3]};
    uint32_t strides[4] = {tensor->strides[0], tensor->strides[1], tensor->strides[2], tensor->strides[3]};

    // Apply permutation: new dimension i gets the shape and stride from old dimension axis[i]
    tensor->shapes[0] = shapes[axis[0]];
    tensor->shapes[1] = shapes[axis[1]];
    tensor->shapes[2] = shapes[axis[2]];
    tensor->shapes[3] = shapes[axis[3]];

    tensor->strides[0] = strides[axis[0]];
    tensor->strides[1] = strides[axis[1]];
    tensor->strides[2] = strides[axis[2]];
    tensor->strides[3] = strides[axis[3]];

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_tensor_copy_ref(struct vkllm_context *context, struct vkllm_tensor *tensor, struct vkllm_tensor **p)
{
    _CHECK_ARGS(context && tensor && p);

    *p = (struct vkllm_tensor *)malloc(sizeof(struct vkllm_tensor));
    struct vkllm_tensor *t = *p;

    strncpy((char *)t->name, tensor->name, sizeof(t->name));
    t->dtype = tensor->dtype;
    for (uint32_t i = 0; i < 4; ++i)
    {
        t->shapes[i] = tensor->shapes[i];
        t->strides[i] = tensor->strides[i];
    }
    t->bytes = tensor->bytes;

    t->device = tensor->device;

    t->data.allocation = tensor->data.allocation;
    t->data.alloc_info = tensor->data.alloc_info;
    t->data.vk_buf = tensor->data.vk_buf;
    t->data.host = tensor->data.host;
    t->data.mapped = tensor->data.mapped;
    t->data.is_ref = true;

    t->access_flags = tensor->access_flags;
    t->pipeline_stage = tensor->pipeline_stage;

    t->op = VKLLM_OP_REF;
    t->pipeline = NULL;
    t->srcs[0] = tensor;
    for (uint32_t i = 1; i < VKLLM_MAX_SRCS; ++i)
    {
        t->srcs[i] = NULL;
    }

    context->stats.tensor_alloc_counts += 1;
    return VKLLM_ERR_OK;
}
