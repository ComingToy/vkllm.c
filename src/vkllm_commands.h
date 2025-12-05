#ifndef __VKLLM_COMMAND_H__
#define __VKLLM_COMMAND_H__

#include <vulkan/vulkan.h>
#include "vkllm_gpu_device.h"

struct vkllm_commands
{
	struct vkllm_gpu_device* device;
	VkCommandBuffer vk_command_buffer;
	VkCommandPool vk_command_pool;
	VkQueue vk_queue;
	VkFence vk_fence;
};

#endif
