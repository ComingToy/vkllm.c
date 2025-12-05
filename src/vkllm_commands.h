#ifndef __VKLLM_COMMAND_H__
#define __VKLLM_COMMAND_H__

#include "vkllm_array.h"
#include "vkllm_gpu_device.h"
#include <vulkan/vulkan.h>

struct vkllm_commands
{
    struct vkllm_gpu_device *device;
    VkCommandBuffer vk_command_buffer;
    VkCommandPool vk_command_pool;
    VkQueue vk_queue;
    VkFence vk_fence;
    uint32_t vk_queue_type;
    struct vkllm_array_ptr *defer_tasks;
};

extern vkllm_err_t vkllm_commands_new(struct vkllm_context *context, struct vkllm_gpu_device *device,
                                      struct vkllm_commands **commands);
extern void vkllm_commands_destroy(struct vkllm_context *context, struct vkllm_commands *commands);

#endif
