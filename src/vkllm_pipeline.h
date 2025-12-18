#ifndef __VKLLM_PIPELINE_H__
#define __VKLLM_PIPELINE_H__

#include "vkllm_array.h"
#include <stdint.h>
#include <vulkan/vulkan.h>

struct vkllm_gpu_device;
struct vkllm_context;
struct vkllm_shader_info
{
    uint32_t binding_count;
    uint32_t push_constant_bytes;
    uint32_t local_x, local_y, local_z;
};

struct vkllm_shader_constants
{
    struct vkllm_array_u8 *data;
    struct vkllm_array_u32 *offsets;
    struct vkllm_array_u32 *sizes;
    size_t bytes;
};

struct vkllm_pipeline
{

    struct vkllm_shader_info shader_info;
    struct vkllm_gpu_device *device;

    VkPipeline vk_pipeline;
    VkShaderModule vk_shader_module;
    VkPipelineLayout vk_pipeline_layout;
    VkDescriptorSetLayout vk_desc_set_layout;
    VkDescriptorPool vk_desc_pool;
    VkDescriptorSet vk_desc_set;
};

extern vkllm_err_t vkllm_shader_constants_new(struct vkllm_shader_constants **constants, uint32_t init_bytes);
extern vkllm_err_t _vkllm_shader_constants_append(struct vkllm_shader_constants *constants, const uint8_t *data,
                                                  uint32_t bytes);

#define vkllm_shader_constants_append(constants, element)                                                              \
    _vkllm_shader_constants_append(constants, (const uint8_t *)&(element), sizeof(element))

#define vkllm_shader_constants_append_n(constants, elements, n)                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        for (uint32_t __i = 0; __i < (n); ++__i)                                                                       \
        {                                                                                                              \
            vkllm_shader_constants_append(constants, (elements)[__i]);                                                 \
        }                                                                                                              \
    } while (0)

extern void vkllm_shader_constants_free(struct vkllm_shader_constants *constants);

extern vkllm_err_t vkllm_pipeline_new(struct vkllm_context *context, struct vkllm_shader_info shader_info,
                                      const uint8_t *spv, const size_t spv_size,
                                      struct vkllm_shader_constants *specializations, struct vkllm_pipeline **pipeline);
extern vkllm_err_t vkllm_pipeline_update_bindings(struct vkllm_context *context, struct vkllm_pipeline *pipeline,
                                                  struct vkllm_array_ptr *bindings, struct vkllm_array_u32 *indices);
extern void vkllm_pipeline_free(struct vkllm_context *context, struct vkllm_pipeline *pipeline);
extern vkllm_err_t vkllm_create_all_pipelines(struct vkllm_context *context);
extern void vkllm_free_all_pipelines(struct vkllm_context *context);
#endif
