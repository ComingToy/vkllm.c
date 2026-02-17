#include "vkllm_pipeline.h"
#include "vkllm_array.h"
#include "vkllm_common.h"
#include "vkllm_tensor.h"

vkllm_err_t vkllm_shader_constants_new(struct vkllm_shader_constants **constants, uint32_t init_bytes)
{
    vkllm_err_t err = VKLLM_ERR_OK;
    _NEW_AND_CHECK(*constants, struct vkllm_shader_constants);

    struct vkllm_shader_constants *p = *constants;
    _CHECK_JUMP(vkllm_array_u8_new(&p->data, init_bytes), err, fail_data);
    _CHECK_JUMP(vkllm_array_u32_new(&p->offsets, init_bytes), err, fail_offsets);
    _CHECK_JUMP(vkllm_array_u32_new(&p->sizes, init_bytes), err, fail_sizes);
    p->bytes = 0;

    return VKLLM_ERR_OK;

fail_sizes:
    vkllm_array_u32_free(p->offsets);
fail_offsets:
    vkllm_array_u8_free(p->data);
fail_data:
    return err;
}

vkllm_err_t _vkllm_shader_constants_append(struct vkllm_shader_constants *constants, const uint8_t *data,
                                           uint32_t bytes)
{
    uint32_t offset = constants->data->used_n;
    for (uint32_t i = 0; i < bytes; ++i)
    {
        _CHECK(vkllm_array_u8_append(constants->data, data[i]));
    }
    _CHECK(vkllm_array_u32_append(constants->offsets, offset));
    _CHECK(vkllm_array_u32_append(constants->sizes, bytes));
    constants->bytes += bytes;
    return VKLLM_ERR_OK;
}

void vkllm_shader_constants_free(struct vkllm_shader_constants *constants)
{
    vkllm_array_u8_free(constants->data);
    vkllm_array_u32_free(constants->offsets);
    vkllm_array_u32_free(constants->sizes);
    free(constants);
}

static vkllm_err_t vkllm_pipeline_init_desc_set_pool(struct vkllm_pipeline *p)
{
    VkDescriptorPoolSize size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .descriptorCount = p->shader_info.binding_count};

    VkDescriptorPoolCreateInfo desc_pool_create_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                        .pNext = NULL,
                                                        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                                        .maxSets = 1,
                                                        .poolSizeCount = 1,
                                                        .pPoolSizes = &size};

    _CHECK_VK(vkCreateDescriptorPool(p->device->vk_dev, &desc_pool_create_info, NULL, &p->vk_desc_pool));
    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_pipeline_create_layout(struct vkllm_pipeline *pipieline)
{

    vkllm_err_t ret = VKLLM_ERR_OK;
    VkDescriptorSetLayoutBinding *bindings;
    _NEW_N_AND_CHECK(bindings, VkDescriptorSetLayoutBinding, pipieline->shader_info.binding_count);

    for (uint32_t i = 0; i < pipieline->shader_info.binding_count; ++i)
    {
        VkDescriptorSetLayoutBinding binding = {.binding = i,
                                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                .descriptorCount = 1, // we don't need array binding
                                                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .pImmutableSamplers = NULL};
        bindings[i] = binding;
    }

    VkDescriptorSetLayoutCreateInfo desc_set_layout_create_Info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .bindingCount = pipieline->shader_info.binding_count,
        .pBindings = bindings};

    _CHECK_VK_JUMP(vkCreateDescriptorSetLayout(pipieline->device->vk_dev, &desc_set_layout_create_Info, NULL,
                                               &pipieline->vk_desc_set_layout),
                   ret, fail);

    VkPushConstantRange range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pipieline->shader_info.push_constant_bytes};

    VkPipelineLayoutCreateInfo layout_create_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                     .pNext = NULL,
                                                     .flags = 0,
                                                     .setLayoutCount = 1,
                                                     .pSetLayouts = &pipieline->vk_desc_set_layout,
                                                     .pushConstantRangeCount =
                                                         pipieline->shader_info.push_constant_bytes > 0 ? 1u : 0u,
                                                     .pPushConstantRanges = &range};

    _CHECK_VK_JUMP(
        vkCreatePipelineLayout(pipieline->device->vk_dev, &layout_create_info, NULL, &pipieline->vk_pipeline_layout),
        ret, fail);

    return VKLLM_ERR_OK;
fail:
    free(bindings);
    return ret;
}

vkllm_err_t vkllm_pipeline_create_shader_module(struct vkllm_pipeline *pipeline, const uint8_t *spv, const size_t size)
{
    VkShaderModuleCreateInfo shader_module_create_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                                          .pNext = NULL,
                                                          .flags = 0,
                                                          .codeSize = size,
                                                          .pCode = (uint32_t *)spv};

    _CHECK_VK(
        vkCreateShaderModule(pipeline->device->vk_dev, &shader_module_create_info, NULL, &pipeline->vk_shader_module));
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_pipeline_new(struct vkllm_context *context, struct vkllm_gpu_device *device,
                               struct vkllm_shader_info shader_info, const uint8_t *spv, const size_t spv_size,
                               struct vkllm_shader_constants *specializations, struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && device && spv && specializations && pipeline);

    const VkPhysicalDeviceLimits *limits = &device->vk_physical_dev.properties.limits;

    if (shader_info.local_x > limits->maxComputeWorkGroupSize[0] ||
        shader_info.local_y > limits->maxComputeWorkGroupSize[1] ||
        shader_info.local_z > limits->maxComputeWorkGroupSize[2])
    {
        log_error("group size out of range. (local_x = %u, local_y = %zu, local_z = %u) > (%u, %u, %u)",
                  shader_info.local_x, shader_info.local_y, shader_info.local_z, limits->maxComputeWorkGroupSize[0],
                  limits->maxComputeWorkGroupSize[1], limits->maxComputeWorkGroupSize[2]);
        return VKLLM_ERR_ARGS;
    }

    struct vkllm_pipeline *p = *pipeline;
    p->shader_info = shader_info;
    p->device = device;

    _CHECK(vkllm_pipeline_create_layout(p));
    _CHECK(vkllm_pipeline_init_desc_set_pool(p));
    _CHECK(vkllm_pipeline_create_shader_module(p, spv, spv_size));

    vkllm_shader_constants_append(specializations, shader_info.local_x);
    vkllm_shader_constants_append(specializations, shader_info.local_y);
    vkllm_shader_constants_append(specializations, shader_info.local_z);

    VkSpecializationMapEntry *specialization_entries = NULL;
    _NEW_N_AND_CHECK(specialization_entries, VkSpecializationMapEntry, specializations->offsets->used_n);

    for (uint32_t i = 0; i < specializations->offsets->used_n - 3; ++i)
    {
        VkSpecializationMapEntry entry = {
            .constantID = i,
            .offset = specializations->offsets->data[i],
            .size = specializations->sizes->data[i],
        };

        specialization_entries[i] = entry;
    }

    for (uint32_t i = 0, k = specializations->offsets->used_n - 3; i < 3; ++i, ++k)
    {
        uint32_t id = 253 + i;
        VkSpecializationMapEntry entry = {
            .constantID = id,
            .offset = specializations->offsets->data[k],
            .size = specializations->sizes->data[k],
        };

        specialization_entries[k] = entry;
    }

    VkSpecializationInfo specialization_info = {.mapEntryCount = specializations->offsets->used_n,
                                                .pMapEntries = specialization_entries,
                                                .dataSize = (size_t)specializations->data->used_n,
                                                .pData = specializations->data->data};

    VkPipelineShaderStageCreateInfo shader_stage_create_info = {.sType =
                                                                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                                .pNext = NULL,
                                                                .flags = 0,
                                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                .module = p->vk_shader_module,
                                                                .pName = "main",
                                                                .pSpecializationInfo = &specialization_info};

    VkComputePipelineCreateInfo pipeline_create_info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                        .pNext = NULL,
                                                        .flags = 0,
                                                        .stage = shader_stage_create_info,
                                                        .layout = p->vk_pipeline_layout,
                                                        .basePipelineHandle = VK_NULL_HANDLE,
                                                        0};

    VkResult vkresult =
        vkCreateComputePipelines(p->device->vk_dev, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &p->vk_pipeline);

    free(specialization_entries);
    if (vkresult != VK_SUCCESS)
    {
        log_error("vkCreateComputePipelines failed: %d", (int)vkresult);
        return VKLLM_ERR_VULKAN;
    }
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_pipeline_update_bindings(struct vkllm_context *context, struct vkllm_pipeline *pipeline,
                                           struct vkllm_array_ptr *bindings, struct vkllm_array_u32 *indices)
{
    VkDescriptorBufferInfo *buffer_infos = NULL;
    _NEW_N_AND_CHECK(buffer_infos, VkDescriptorBufferInfo, bindings->used_n);
    if (indices && (bindings->used_n != indices->used_n))
    {
        log_error("input bindings.size != indices.size");
        return VKLLM_ERR_ARGS;
    }

    for (uint32_t i = 0; i < bindings->used_n; ++i)
    {
        struct vkllm_tensor *binding = (struct vkllm_tensor *)bindings->data[i];
        buffer_infos[i].buffer = binding->data.vk_buf;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = VK_WHOLE_SIZE;
    }

    VkWriteDescriptorSet *writers = (VkWriteDescriptorSet *)malloc(sizeof(VkWriteDescriptorSet) * bindings->used_n);
    if (!writers)
    {
        free(buffer_infos);
        return VKLLM_ERR_ALLOC;
    }

    for (uint32_t i = 0; i < bindings->used_n; ++i)
    {
        VkWriteDescriptorSet writer = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                       .pNext = NULL,
                                       .dstSet = pipeline->vk_desc_set,
                                       .dstBinding = indices ? indices->data[i] : i,
                                       .dstArrayElement = 0,
                                       .descriptorCount = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .pImageInfo = NULL,
                                       .pBufferInfo = &buffer_infos[i],
                                       .pTexelBufferView = NULL};
        writers[i] = writer;
    }

    vkUpdateDescriptorSets(pipeline->device->vk_dev, bindings->used_n, writers, 0, NULL);
    free(buffer_infos);
    free(writers);
    return VKLLM_ERR_OK;
}

void vkllm_pipeline_free(struct vkllm_context *context, struct vkllm_pipeline *pipeline)
{
    vkDestroyPipeline(pipeline->device->vk_dev, pipeline->vk_pipeline, NULL);
    vkDestroyShaderModule(pipeline->device->vk_dev, pipeline->vk_shader_module, NULL);
    vkDestroyPipelineLayout(pipeline->device->vk_dev, pipeline->vk_pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(pipeline->device->vk_dev, pipeline->vk_desc_set_layout, NULL);
    vkDestroyDescriptorPool(pipeline->device->vk_dev, pipeline->vk_desc_pool, NULL);
    free(pipeline);
}
