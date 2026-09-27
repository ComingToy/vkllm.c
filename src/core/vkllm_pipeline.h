#ifndef __VKLLM_PIPELINE_H__
#define __VKLLM_PIPELINE_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "vkllm_array.h"
#include <stdint.h>
#include <string.h>
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
    VkQueryPool vk_query_pool;
    char *name[];
};

// Fixed-capacity push-constant block, filled with plain stores (no heap,
// no realloc) and submitted to the command buffer in one vkCmdPushConstants.
// Vulkan guarantees >= 128 bytes of push constants; 128 covers every op.
#define VKLLM_PUSH_CONSTANTS_MAX 128
struct vkllm_push_constants
{
    uint32_t bytes;
    uint8_t data[VKLLM_PUSH_CONSTANTS_MAX];
};

static inline void vkllm_push_constants_reset(struct vkllm_push_constants *pc)
{
    pc->bytes = 0;
}

static inline vkllm_err_t vkllm_push_constants_append(struct vkllm_push_constants *pc, const void *src, uint32_t bytes)
{
    if (pc->bytes + bytes > VKLLM_PUSH_CONSTANTS_MAX)
        return VKLLM_ERR_ARGS;
    memcpy(pc->data + pc->bytes, src, bytes);
    pc->bytes += bytes;
    return VKLLM_ERR_OK;
}

// Typed appenders matching the old vkllm_shader_constants_append call sites.
static inline vkllm_err_t vkllm_push_u32(struct vkllm_push_constants *pc, uint32_t v)
{
    return vkllm_push_constants_append(pc, &v, sizeof(v));
}

static inline vkllm_err_t vkllm_push_i32(struct vkllm_push_constants *pc, int32_t v)
{
    return vkllm_push_constants_append(pc, &v, sizeof(v));
}

static inline vkllm_err_t vkllm_push_f32(struct vkllm_push_constants *pc, float v)
{
    return vkllm_push_constants_append(pc, &v, sizeof(v));
}

static inline vkllm_err_t vkllm_push_u32_n(struct vkllm_push_constants *pc, const uint32_t *vs, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        vkllm_err_t err = vkllm_push_u32(pc, vs[i]);
        if (err != VKLLM_ERR_OK)
            return err;
    }
    return VKLLM_ERR_OK;
}

extern vkllm_err_t vkllm_shader_constants_new(struct vkllm_shader_constants **constants, uint32_t init_bytes);
extern vkllm_err_t vkllm_shader_constants_copy(const struct vkllm_shader_constants *src,
                                               struct vkllm_shader_constants **dst);
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

extern vkllm_err_t vkllm_pipeline_new(struct vkllm_context *context, const char *name,
                                      struct vkllm_shader_info shader_info, const uint8_t *spv, const size_t spv_size,
                                      struct vkllm_shader_constants *specializations, struct vkllm_pipeline **pipeline);
extern vkllm_err_t vkllm_pipeline_update_bindings(struct vkllm_context *context, struct vkllm_pipeline *pipeline,
                                                  struct vkllm_array_ptr *bindings, struct vkllm_array_u32 *indices);
extern void vkllm_pipeline_free(struct vkllm_context *context, struct vkllm_pipeline *pipeline);
extern vkllm_err_t vkllm_create_all_pipelines(struct vkllm_context *context);
extern void vkllm_free_all_pipelines(struct vkllm_context *context);
extern vkllm_err_t vkllm_pipeline_alloc_desc_set(struct vkllm_context *context, struct vkllm_pipeline *pipeline,
                                                 VkDescriptorSet *vk_desc_set);
extern vkllm_err_t vkllm_pipeline_query_exec_time(struct vkllm_context *context, struct vkllm_pipeline *pipeline,
                                                  uint64_t *cost);

#ifdef __cplusplus
}
#endif
#endif
