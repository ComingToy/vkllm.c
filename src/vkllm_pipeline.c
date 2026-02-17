#include "vkllm_pipeline.h"
#include "vkllm_add_shaders.h"
#include "vkllm_array.h"
#include "vkllm_common.h"
#include "vkllm_context.h"
#include "vkllm_dtypes.h"
#include "vkllm_embedding_shaders.h"
#include "vkllm_errors.h"
#include "vkllm_gpu_device.h"
#include "vkllm_matmul_shaders.h"
#include "vkllm_rmsnorm_shaders.h"
#include "vkllm_tensor.h"
#include <stdlib.h>
#include <string.h>

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

vkllm_err_t vkllm_shader_constants_copy(const struct vkllm_shader_constants *src, struct vkllm_shader_constants **dst)
{
    vkllm_err_t err = VKLLM_ERR_OK;

    _CHECK_ARGS(src);
    _NEW_AND_CHECK(*dst, struct vkllm_shader_constants);

    struct vkllm_shader_constants *p = *dst;
    _CHECK_JUMP(vkllm_array_u8_copy(src->data, &p->data), err, fail_copy);
    _CHECK_JUMP(vkllm_array_u32_copy(src->offsets, &p->offsets), err, fail_copy);
    _CHECK_JUMP(vkllm_array_u32_copy(src->sizes, &p->sizes), err, fail_copy);
    p->bytes = src->bytes;

    return err;

fail_copy:
    vkllm_shader_constants_free(p);
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

    VkDescriptorSetAllocateInfo desc_set_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                       .pNext = NULL,
                                                       .descriptorPool = p->vk_desc_pool,
                                                       .descriptorSetCount = 1,
                                                       .pSetLayouts = &p->vk_desc_set_layout};

    _CHECK_VK(vkAllocateDescriptorSets(p->device->vk_dev, &desc_set_alloc_info, &p->vk_desc_set));
    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_pipeline_create_layout(struct vkllm_pipeline *pipeline)
{

    vkllm_err_t ret = VKLLM_ERR_OK;
    VkDescriptorSetLayoutBinding *bindings;
    _NEW_N_AND_CHECK(bindings, VkDescriptorSetLayoutBinding, pipeline->shader_info.binding_count);

    for (uint32_t i = 0; i < pipeline->shader_info.binding_count; ++i)
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
        .bindingCount = pipeline->shader_info.binding_count,
        .pBindings = bindings};

    _CHECK_VK_JUMP(vkCreateDescriptorSetLayout(pipeline->device->vk_dev, &desc_set_layout_create_Info, NULL,
                                               &pipeline->vk_desc_set_layout),
                   ret, fail);

    free(bindings);

    VkPushConstantRange range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pipeline->shader_info.push_constant_bytes};

    VkPipelineLayoutCreateInfo layout_create_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                     .pNext = NULL,
                                                     .flags = 0,
                                                     .setLayoutCount = 1,
                                                     .pSetLayouts = &pipeline->vk_desc_set_layout,
                                                     .pushConstantRangeCount =
                                                         pipeline->shader_info.push_constant_bytes > 0 ? 1u : 0u,
                                                     .pPushConstantRanges = &range};

    _CHECK_VK_JUMP(
        vkCreatePipelineLayout(pipeline->device->vk_dev, &layout_create_info, NULL, &pipeline->vk_pipeline_layout), ret,
        fail);

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

vkllm_err_t vkllm_pipeline_new(struct vkllm_context *context, const char *name, struct vkllm_shader_info shader_info,
                               const uint8_t *spv, const size_t spv_size,
                               struct vkllm_shader_constants *specializations, struct vkllm_pipeline **pipeline)
{
    struct vkllm_gpu_device *device = context->device;
    _CHECK_ARGS(context && device && spv && pipeline);

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

    size_t name_len = name ? strlen(name) : 0;
    *pipeline = (struct vkllm_pipeline *)malloc(sizeof(struct vkllm_pipeline) + name_len);

    struct vkllm_pipeline *p = *pipeline;
    p->shader_info = shader_info;
    p->device = device;
    strncpy((char *)p->name, name, name_len);

    // FIXME: leak pipeline
    _CHECK(vkllm_pipeline_create_layout(p));
    _CHECK(vkllm_pipeline_init_desc_set_pool(p));
    _CHECK(vkllm_pipeline_create_shader_module(p, spv, spv_size));

    struct vkllm_shader_constants *local_specializations = NULL;
    if (specializations)
    {
        _CHECK(vkllm_shader_constants_copy(specializations, &local_specializations));
    }
    else
    {
        _CHECK(vkllm_shader_constants_new(&local_specializations, 24));
    }

    vkllm_shader_constants_append(local_specializations, shader_info.local_x);
    vkllm_shader_constants_append(local_specializations, shader_info.local_y);
    vkllm_shader_constants_append(local_specializations, shader_info.local_z);

    VkSpecializationMapEntry *specialization_entries = NULL;
    _NEW_N_AND_CHECK(specialization_entries, VkSpecializationMapEntry, local_specializations->offsets->used_n);

    for (uint32_t i = 0; i < local_specializations->offsets->used_n - 3; ++i)
    {
        VkSpecializationMapEntry entry = {
            .constantID = i,
            .offset = local_specializations->offsets->data[i],
            .size = local_specializations->sizes->data[i],
        };

        specialization_entries[i] = entry;
    }

    for (uint32_t i = 0, k = local_specializations->offsets->used_n - 3; i < 3; ++i, ++k)
    {
        uint32_t id = 253 + i;
        VkSpecializationMapEntry entry = {
            .constantID = id,
            .offset = local_specializations->offsets->data[k],
            .size = local_specializations->sizes->data[k],
        };

        specialization_entries[k] = entry;
    }

    VkSpecializationInfo specialization_info = {.mapEntryCount = local_specializations->offsets->used_n,
                                                .pMapEntries = specialization_entries,
                                                .dataSize = (size_t)local_specializations->data->used_n,
                                                .pData = local_specializations->data->data};

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
    vkllm_shader_constants_free(local_specializations);
    if (vkresult != VK_SUCCESS)
    {
        log_error("vkCreateComputePipelines failed: %d", (int)vkresult);
        return VKLLM_ERR_VULKAN;
    }

    VkQueryPoolCreateInfo query_pool_create_info = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
        .pipelineStatistics = 0,
    };

    _CHECK_VK(vkCreateQueryPool(p->device->vk_dev, &query_pool_create_info, NULL, &p->vk_query_pool));
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
    if (!pipeline)
        return;
    vkDestroyPipeline(pipeline->device->vk_dev, pipeline->vk_pipeline, NULL);
    vkDestroyShaderModule(pipeline->device->vk_dev, pipeline->vk_shader_module, NULL);
    vkDestroyPipelineLayout(pipeline->device->vk_dev, pipeline->vk_pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(pipeline->device->vk_dev, pipeline->vk_desc_set_layout, NULL);
    vkDestroyDescriptorPool(pipeline->device->vk_dev, pipeline->vk_desc_pool, NULL);
    vkDestroyQueryPool(pipeline->device->vk_dev, pipeline->vk_query_pool, NULL);
    free(pipeline);
}

vkllm_err_t vkllm_pipeline_query_exec_time(struct vkllm_context *context, struct vkllm_pipeline *pipeline,
                                           uint64_t *cost)
{
    _CHECK_ARGS(context->device->support_query_timestamp && cost);

    uint64_t time_stamps[2] = {0, 0};
    vkGetQueryPoolResults(context->device->vk_dev, pipeline->vk_query_pool, 0, 2, sizeof(time_stamps), time_stamps,
                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    float delta =
        (time_stamps[1] - time_stamps[0]) * context->device->vk_physical_dev.properties.limits.timestampPeriod;
    *cost = (uint64_t)delta;
    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_create_all_add_pipeline(struct vkllm_context *context)
{
#define _CREATE_ADD_PIPELINE(_tag)                                                                                     \
    _CHECK(vkllm_pipeline_new(context, "add_" #_tag, shader_info, _vkllm_add_##_tag##_spv(),                           \
                              _vkllm_add_##_tag##_size(), NULL, &context->pipelines.add._tag))
    _CHECK_ARGS(context);
    struct vkllm_shader_info shader_info = {
        .binding_count = 3, .push_constant_bytes = sizeof(uint32_t) * 8, .local_x = 512, .local_y = 1, .local_z = 1};

    context->pipelines.add.f16f16f16 = NULL;
    context->pipelines.add.f16f32f32 = NULL;
    context->pipelines.add.f16f16f32 = NULL;
    context->pipelines.add.f32f32f32 = NULL;

    if (context->device->support_16bit_storage)
    {
        _CREATE_ADD_PIPELINE(f16f32f32);
        if (context->device->support_fp16_arithmetic)
        {
            _CREATE_ADD_PIPELINE(f16f16f16);
            _CREATE_ADD_PIPELINE(f16f16f32);
        }
    }

    _CREATE_ADD_PIPELINE(f32f32f32);

#undef _CREATE_ADD_PIPELINE
    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_create_embedding_pipeline(struct vkllm_context *context)
{
    _CHECK_ARGS(context);
    struct vkllm_shader_info shader_info = {
        .binding_count = 3, .push_constant_bytes = sizeof(uint32_t) * 25, .local_x = 512, .local_y = 1, .local_z = 1};

    context->pipelines.embedding.f16 = NULL;
    context->pipelines.embedding.f32 = NULL;

    if (context->device->support_16bit_storage)
    {
        _CHECK(vkllm_pipeline_new(context, "pipeline_embedding_f16", shader_info, _vkllm_embedding_f16_spv(),
                                  _vkllm_embedding_f16_size(), NULL, &context->pipelines.embedding.f16));
    }

    _CHECK(vkllm_pipeline_new(context, "pipeline_embedding_f32", shader_info, _vkllm_embedding_f32_spv(),
                              _vkllm_embedding_f32_size(), NULL, &context->pipelines.embedding.f32));
    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_create_rmsnorm_pipeline(struct vkllm_context *context)
{
    _CHECK_ARGS(context);

    struct vkllm_shader_info shader_info = {
        .binding_count = 3,
        .push_constant_bytes = sizeof(uint32_t) * 8 * 2 + sizeof(float) * 2,
        .local_x = 512,
        .local_y = 1,
        .local_z = 1,
    };

    context->pipelines.rmsnorm.f16f32f32 = NULL;

    if (context->device->support_16bit_storage)
    {
        _CHECK(vkllm_pipeline_new(context, "pipeline_rmsnorm_f16f32f32", shader_info, _vkllm_rmsnorm_f16f32f32_spv(),
                                  _vkllm_rmsnorm_f16f32f32_size(), NULL, &context->pipelines.rmsnorm.f16f32f32));
        _CHECK(vkllm_pipeline_new(context, "pipeline_rmsnorm_f16f32f16", shader_info, _vkllm_rmsnorm_f16f32f16_spv(),
                                  _vkllm_rmsnorm_f16f32f16_size(), NULL, &context->pipelines.rmsnorm.f16f32f16));
    }

    _CHECK(vkllm_pipeline_new(context, "pipeline_rmsnorm_f32f32f32", shader_info, _vkllm_rmsnorm_f32f32f32_spv(),
                              _vkllm_rmsnorm_f32f32f32_size(), NULL, &context->pipelines.rmsnorm.f32f32f32));

    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_create_matmul_pipelines(struct vkllm_context *context)
{
    _CHECK_ARGS(context);

    struct vkllm_shader_info shader_info = {
        .binding_count = 3,
        .push_constant_bytes = sizeof(uint32_t) * 6,
        .local_x = 16,
        .local_y = 16,
        .local_z = 1,
    };

    context->pipelines.matmul.f32f32f32 = NULL;
    context->pipelines.matmul.f16f32f32 = NULL;
    context->pipelines.matmul.f16f16f32 = NULL;

    _CHECK(vkllm_pipeline_new(context, "matmul_f32f32f32", shader_info, _vkllm_matmul_f32f32f32_spv(),
                              _vkllm_matmul_f32f32f32_size(), NULL, &context->pipelines.matmul.f32f32f32));

    if (context->device->support_16bit_storage)
    {
        _CHECK(vkllm_pipeline_new(context, "matmul_f16f32f32", shader_info, _vkllm_matmul_f16f32f32_spv(),
                                  _vkllm_matmul_f16f32f32_size(), NULL, &context->pipelines.matmul.f16f32f32));

        if (context->device->support_fp16_arithmetic)
        {

            _CHECK(vkllm_pipeline_new(context, "matmul_f16f16f32", shader_info, _vkllm_matmul_f16f16f32_spv(),
                                      _vkllm_matmul_f16f16f32_size(), NULL, &context->pipelines.matmul.f16f32f32));
        }
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_create_all_pipelines(struct vkllm_context *context)
{
    _CHECK(vkllm_create_all_add_pipeline(context));
    _CHECK(vkllm_create_embedding_pipeline(context));
    _CHECK(vkllm_create_rmsnorm_pipeline(context));
    _CHECK(vkllm_create_matmul_pipelines(context));
    return VKLLM_ERR_OK;
}

void vkllm_free_all_pipelines(struct vkllm_context *context)
{
    vkllm_pipeline_free(context, context->pipelines.add.f16f16f16);
    vkllm_pipeline_free(context, context->pipelines.add.f16f32f32);
    vkllm_pipeline_free(context, context->pipelines.add.f32f32f32);

    vkllm_pipeline_free(context, context->pipelines.embedding.f16);
    vkllm_pipeline_free(context, context->pipelines.embedding.f32);

    vkllm_pipeline_free(context, context->pipelines.rmsnorm.f16f32f32);
    vkllm_pipeline_free(context, context->pipelines.rmsnorm.f16f32f16);
    vkllm_pipeline_free(context, context->pipelines.rmsnorm.f32f32f32);

    vkllm_pipeline_free(context, context->pipelines.matmul.f32f32f32);
    vkllm_pipeline_free(context, context->pipelines.matmul.f16f32f32);
    vkllm_pipeline_free(context, context->pipelines.matmul.f16f16f32);
}
#undef vkllm_free_op_pipelines
#undef _vkllm_free_op_pipeline
#undef _member
