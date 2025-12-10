#ifndef __VKLLM_GPU_DEVICE_H__
#define __VKLLM_GPU_DEVICE_H__

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "vk_mem_alloc.h"
#include "vkllm_errors.h"

struct vkllm_context;
struct vkllm_vk_physical_dev
{
    uint32_t id;
    VkPhysicalDevice dev;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceProperties properties;
    VkExtensionProperties *ext_properties;
    uint32_t n_ext_properties;
    VkPhysicalDeviceMemoryProperties mem_properties;
    VkQueueFamilyProperties *queue_family_properties;
    uint32_t n_queue_family_properties;
    VkPhysicalDeviceSubgroupProperties subgroup_properties;
    VkPhysicalDeviceShaderFloat16Int8Features feat_shader_fp16_int8;
    VkPhysicalDevice16BitStorageFeatures feat_16bit_storage;
    VkPhysicalDevice8BitStorageFeatures feat_8bit_storage;
    VkPhysicalDeviceProperties2 properties2;
    VkPhysicalDeviceFeatures2 features2;
};

struct vkllm_gpu_device
{
    uint32_t api_version;
    VkInstance vk_instance;
    VkDevice vk_dev;
    struct vkllm_vk_physical_dev vk_physical_dev;
    bool support_descriptor_templ_update;
    bool support_16bit_storage;
    bool support_8bit_storage;
    bool support_fp16_arithmetic;
    bool support_int8_arithmetic;
    bool support_pipeline_statistics;
    int subgroup_size;
    VmaAllocator vma_allocator;
};

extern vkllm_err_t vkllm_gpu_device_new(struct vkllm_context *context, uint32_t id);
extern vkllm_err_t vkllm_gpu_device_require_queue(struct vkllm_context *context, VkQueueFlagBits flags, uint32_t *type);
extern void vkllm_gpu_device_free(struct vkllm_context *context);
#endif
