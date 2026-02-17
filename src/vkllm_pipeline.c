#include "vkllm_pipeline.h"
#include "vkllm_common.h"

vkllm_err_t vkllm_shader_constants_new(struct vkllm_shader_constants **constants, uint32_t init_bytes)
{
    _NEW_AND_CHECK(*constants, struct vkllm_shader_constants);
    struct vkllm_shader_constants *p = *constants;
    _CHECK(vkllm_array_u8_new(&p->data, init_bytes));
    _CHECK(vkllm_array_u32_new(&p->offsets, init_bytes));
    _CHECK(vkllm_array_u32_new(&p->sizes, init_bytes));
    return VKLLM_ERR_OK;
}

vkllm_err_t _vkllm_shader_constants_append(struct vkllm_shader_constants *constants, uint8_t *data, uint32_t bytes)
{
    uint32_t offset = constants->data->used_n;
    for (uint32_t i = 0; i < bytes; ++i)
    {
        _CHECK(vkllm_array_u8_append(constants->data, data[i]));
    }
    _CHECK(vkllm_array_u32_append(constants->offsets, offset));
    _CHECK(vkllm_array_u32_append(constants->sizes, bytes));
    return VKLLM_ERR_OK;
}

void vkllm_shader_constants_free(struct vkllm_shader_constants *constants)
{
    vkllm_array_u8_free(constants->data);
    vkllm_array_u32_free(constants->offsets);
    vkllm_array_u32_free(constants->sizes);
    free(constants);
}

static vkllm_err_t vkllm_pipeline_init_desc_set(struct vkllm_pipeline *p)
{
    VkDescriptorPoolCreateInfo desc_pool_create_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                        .pNext = NULL,
                                                        .flags = 0,
                                                        .maxSets = 1,
                                                        .poolSizeCount = 0,
                                                        .pPoolSizes = NULL};
}

vkllm_err_t vkllm_pipeline_new(struct vkllm_context *context, struct vkllm_gpu_device *device, const uint8_t *spv,
                               const size_t spv_size, const struct vkllm_shader_constants *specializations,
                               struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && device && spv && specializations && pipeline);

    struct vkllm_pipeline *p = *pipeline;

    VkPipelineShaderStageCreateInfo shader_stage_create_info;
    VkComputePipelineCreateInfo pipeline_create_info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                        .pNext = NULL,
                                                        .flags = 0,
                                                        .stage = shader_stage_create_info,
                                                        .layout = p->vk_pipeline_layout,
                                                        .basePipelineHandle = VK_NULL_HANDLE,
                                                        0};
    return VKLLM_ERR_OK;
}
