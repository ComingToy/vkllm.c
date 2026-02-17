#ifndef __VKLLM_COMMANDS_H__
#define __VKLLM_COMMANDS_H__

#include "vkllm_array.h"
#include <vulkan/vulkan.h>

struct vkllm_gpu_device;
struct vkllm_tensor;
struct vkllm_context;
struct vkllm_pipeline;
struct vkllm_shader_constants;
struct vkllm_commands_task
{
    vkllm_err_t (*func)(void *);
    struct vkllm_context *context;
    void *priv;
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

extern vkllm_err_t vkllm_commands_new(struct vkllm_context *context, struct vkllm_commands **commands);
extern void vkllm_commands_free(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_begin(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_end(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_upload(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor, const uint8_t *data, size_t bytes);
extern vkllm_err_t vkllm_commands_download(struct vkllm_context *context, struct vkllm_commands *commands,
                                           struct vkllm_tensor *tensor, uint8_t *data, size_t bytes);
extern void vkllm_commands_sync_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                       struct vkllm_tensor *tensor, VkAccessFlagBits dst_access,
                                       VkPipelineStageFlagBits dst_stage);
extern vkllm_err_t vkllm_commands_pipeline(struct vkllm_context *context, struct vkllm_commands *commands,
                                           struct vkllm_pipeline *pipeline, struct vkllm_array_ptr *bindings,
                                           struct vkllm_array_u32 *indices, struct vkllm_shader_constants *constants,
                                           uint32_t group_x, uint32_t group_y, uint32_t group_z);
extern vkllm_err_t vkllm_commands_submit(struct vkllm_context *context, struct vkllm_commands *commands);
extern vkllm_err_t vkllm_commands_wait_exec(struct vkllm_context *context, struct vkllm_commands *commands);

extern void __vkllm_commands_sync_tensor(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor, VkAccessFlagBits dst_access,
                                         VkPipelineStageFlagBits dst_stage);
#endif
