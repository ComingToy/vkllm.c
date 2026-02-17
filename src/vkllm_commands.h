#ifndef __VKLLM_COMMAND_H__
#define __VKLLM_COMMAND_H__

#include "vkllm_array.h"
#include "vkllm_gpu_device.h"
#include "vkllm_tensor.h"
#include <vulkan/vulkan.h>

struct vkllm_commands_task
{
    void (*func)(void *);
    void *args;
};

VKLLM_DEF_ARRAY(commands_task, struct vkllm_commands_task);

struct vkllm_commands
{
    struct vkllm_gpu_device *device;
    VkCommandBuffer vk_command_buffer;
    VkCommandPool vk_command_pool;
    VkQueue vk_queue;
    VkFence vk_fence;
    uint32_t vk_queue_type;
    struct vkllm_array_commands_task *defer_tasks;
};

extern vkllm_err_t vkllm_commands_new(struct vkllm_context *context, struct vkllm_gpu_device *device,
                                      struct vkllm_commands **commands);
extern void vkllm_commands_free(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_begin(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_end(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_upload(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor, const uint8_t *data, size_t bytes);
extern vkllm_err_t vkllm_commands_download(struct vkllm_context *context, struct vkllm_commands *commands,
                                           struct vkllm_tensor *tensor, uint8_t *data, size_t bytes);

extern void __vkllm_commands_sync_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor, VkAccessFlagBits dst_access,
                                         VkPipelineStageFlagBits dst_stage);
#endif
