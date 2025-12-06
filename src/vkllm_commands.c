#include "vkllm_commands.h"
#include "log.h"
#include "src/vkllm_array.h"
#include "src/vkllm_common.h"
#include "src/vkllm_gpu_device.h"
#include "src/vkllm_tensor.h"
#include <cstring>
#include <vulkan/vulkan.h>

static vkllm_err_t vkllm_create_command_buffer(struct vkllm_commands *commands)
{
    VkCommandPoolCreateInfo command_pool_create_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                                        .pNext = NULL,
                                                        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                                        .queueFamilyIndex = commands->vk_queue_type};
    VkResult ret =
        vkCreateCommandPool(commands->device->vk_dev, &command_pool_create_info, NULL, &commands->vk_command_pool);
    if (ret != VK_SUCCESS)
    {
        log_error("vkCreateCommandPool failed: %d", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    VkCommandBufferAllocateInfo command_buffer_alloc_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                             .pNext = NULL,
                                                             .commandPool = commands->vk_command_pool,
                                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                             .commandBufferCount = 1};
    ret = vkAllocateCommandBuffers(commands->device->vk_dev, &command_buffer_alloc_info, &commands->vk_command_buffer);
    if (ret != VK_SUCCESS)
    {
        log_error("vkAllocateCommandBuffers failed: %d\n", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_commands_init_queue(struct vkllm_context *context, struct vkllm_commands *commands)
{
    _CHECK(vkllm_gpu_device_require_queue(context, commands->device, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
                                          &commands->vk_queue_type));
    vkGetDeviceQueue(commands->device->vk_dev, commands->vk_queue_type, 0, &commands->vk_queue);

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_commands_new(struct vkllm_context *context, struct vkllm_gpu_device *device,
                               struct vkllm_commands **commands)
{
    if (!context || !device || !commands)
    {
        log_error("input error. context is NULL: %s, device is NULL: %s, commands is NULL: %s", BOOL_S(context),
                  BOOL_S(device), BOOL_S(commands));
    }

    _NEW_AND_CHECK(*commands, struct vkllm_commands);

    struct vkllm_commands *p = *commands;
    vkllm_array_commands_task_new(&p->defer_tasks, 128);
    p->device = device;

    VkFenceCreateInfo fence_create_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VkResult ret = vkCreateFence(p->device->vk_dev, &fence_create_info, NULL, &p->vk_fence);
    if (ret != VK_SUCCESS)
    {
        log_error("vkCreateFence failed: %d\n", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    _CHECK(vkllm_commands_init_queue(context, p));
    _CHECK(vkllm_create_command_buffer(p));
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_commands_begin(struct vkllm_context *context, struct vkllm_commands *commands)
{
    VkCommandBufferBeginInfo buffer_begin_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                  .pNext = NULL,
                                                  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                                  NULL};
    _CHECK_VK(vkBeginCommandBuffer(commands->vk_command_buffer, &buffer_begin_info));
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_commands_end(struct vkllm_context *context, struct vkllm_commands *commands)
{
    _CHECK_VK(vkEndCommandBuffer(commands->vk_command_buffer));
    return VKLLM_ERR_OK;
}

void __vkllm_commands_sync_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                  struct vkllm_tensor *tensor, VkAccessFlagBits dst_access,
                                  VkPipelineStageFlagBits dst_stage)
{
    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = NULL,
        .srcAccessMask = tensor->access_flags,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = tensor->data.vk_buf,
        .offset = 0,
        .size = tensor->bytes,
    };

    vkCmdPipelineBarrier(commands->vk_command_buffer, tensor->pipeline_stage, dst_stage, 0, 0, NULL, 1, &barrier, 0,
                         NULL);
}

vkllm_err_t vkllm_commands_upload(struct vkllm_context *context, struct vkllm_commands *commands,
                                  struct vkllm_tensor *tensor, const uint8_t *data, size_t bytes)
{
    _CHECK_ARGS(context && commands && tensor && data);
    _CHECK_ARGS(bytes <= tensor->bytes);
    if (tensor->data.mapped)
    {
        memcpy(tensor->data.host, data, bytes);
        vkllm_tensor_invalid_cache(context, tensor);
        tensor->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        tensor->pipeline_stage = VK_PIPELINE_STAGE_HOST_BIT;
        return VKLLM_ERR_OK;
    }
    else if (bytes < 65536 && bytes % 4 == 0)
    {
        vkCmdUpdateBuffer(commands->vk_command_buffer, tensor->data.vk_buf, 0, bytes, (const void *)data);
        tensor->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        tensor->pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        return VKLLM_ERR_OK;
    }

    struct vkllm_tensor *staging = NULL;
    _CHECK(vkllm_tensor_new_staging(context, tensor, &staging));

    memcpy(staging->data.host, data, bytes);
    tensor->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    tensor->pipeline_stage = VK_PIPELINE_STAGE_HOST_BIT;

    __vkllm_commands_sync_tensor(context, commands, staging, VK_ACCESS_TRANSFER_READ_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferCopy region = {0, 0, bytes};
    vkCmdCopyBuffer(commands->vk_command_buffer, staging->data.vk_buf, tensor->data.vk_buf, 1, &region);

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_commands_download(struct vkllm_context *context, struct vkllm_commands *commands,
                                    struct vkllm_tensor *tensor, uint8_t *data, size_t bytes)
{
    _CHECK_ARGS(context && commands && tensor && data);
    _CHECK_ARGS(bytes >= tensor->bytes);

    if (tensor->data.mapped)
    {
        __vkllm_commands_sync_tensor(context, commands, tensor, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        vkllm_tensor_flush_cache(context, tensor);
        memcpy(data, tensor->data.host, bytes);
        return VKLLM_ERR_OK;
    }

    __vkllm_commands_sync_tensor(context, commands, tensor, VK_ACCESS_TRANSFER_READ_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    struct vkllm_tensor *staging;
    _CHECK(vkllm_tensor_new_staging(context, tensor, &staging));

    VkBufferCopy region = {.srcOffset = 0, .dstOffset = 0, .size = tensor->bytes};
    vkCmdCopyBuffer(commands->vk_command_buffer, tensor->data.vk_buf, staging->data.vk_buf, 1, &region);
    staging->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
    staging->pipeline_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    __vkllm_commands_sync_tensor(context, commands, staging, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT);
    vkllm_tensor_flush_cache(context, staging);
    memcpy(data, staging->data.host, bytes);
}

void vkllm_commands_free(struct vkllm_context *context, struct vkllm_commands *commands)
{
    vkFreeCommandBuffers(commands->device->vk_dev, commands->vk_command_pool, 1, &commands->vk_command_buffer);
    vkDestroyCommandPool(commands->device->vk_dev, commands->vk_command_pool, NULL);
    vkDestroyFence(commands->device->vk_dev, commands->vk_fence, NULL);
    vkllm_array_commands_task_free(commands->defer_tasks);
    free(commands);
}
