#ifndef __VKLLM_GPU_DEVICE_H__
#define __VKLLM_GPU_DEVICE_H__

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "vkllm_context.h"
#include "vkllm_errors.h"

struct vkllm_gpu_device {
    int id;
    uint32_t api_version;
    VkInstance instance;
    bool support_16bit_storage;
    bool support_8bit_storage;
    bool support_fp16_arithmetic;
    bool support_int8_arithmetic;
    bool support_pipeline_statistics;
    int subgroup_size;
};

extern vkllm_err_t new_gpu_device(struct vkllm_context* context, int id,
				  struct vkllm_gpu_device** ppdev);
#endif
