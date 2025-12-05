#include "vkllm_commands.h"
#include "log.h"
#include "src/vkllm_array.h"
#include "src/vkllm_common.h"
#include "src/vkllm_gpu_device.h"
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
    vkllm_array_ptr_new(&p->defer_tasks, 128);
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

void vkllm_commands_destroy(struct vkllm_context *context, struct vkllm_commands *commands)
{
    vkFreeCommandBuffers(commands->device->vk_dev, commands->vk_command_pool, 1, &commands->vk_command_buffer);
    vkDestroyCommandPool(commands->device->vk_dev, commands->vk_command_pool, NULL);
    vkDestroyFence(commands->device->vk_dev, commands->vk_fence, NULL);
    vkllm_array_ptr_free(commands->defer_tasks);
    free(commands);
}
