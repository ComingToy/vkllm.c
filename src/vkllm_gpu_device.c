#include "vkllm_gpu_device.h"
#include "vkllm_context.h"

#include <log.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkllm_common.h"
#include "vulkan/vulkan_core.h"

// #define __VKLLM_DEBUG__ 1

#ifdef __VKLLM_DEBUG__
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,    // 消息严重级别
              VkDebugUtilsMessageTypeFlagsEXT messageType,               // 消息类型
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, // 回调数据
              void *pUserData)
{
    fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}
#endif

static int create_instance(struct vkllm_gpu_device *pdev)
{
    VkResult ret = vkEnumerateInstanceVersion(&pdev->api_version);
    if (ret != VK_SUCCESS)
    {
        log_error("call vkEnumerateInstanceVersion failed: %d", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    VkApplicationInfo app_info = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                  .pNext = NULL,
                                  .pApplicationName = "vkllm.c",
                                  .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                  .pEngineName = NULL,
                                  .engineVersion = 0,
                                  .apiVersion = pdev->api_version};

    const char *enable_layers[] = {
#ifdef __VKLLM_DEBUG__
        "VK_LAYER_KHRONOS_validation",
#endif
    };

    const char *exts[] = {
#ifdef __APPLE__
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
#ifdef __VKLLM_DEBUG__
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
    };

    VkValidationFeatureEnableEXT enabledFeatures[] = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};

    VkValidationFeaturesEXT validationFeatures = {.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
                                                  .enabledValidationFeatureCount = 1,
                                                  .pEnabledValidationFeatures = enabledFeatures};

    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = &validationFeatures,
#if __APPLE__
        .flags = VK_KHR_portability_enumeration,
#else
        .flags = 0,
#endif
        .pApplicationInfo = &app_info,
        .enabledLayerCount = sizeof(enable_layers) / sizeof(const char *),
        .ppEnabledLayerNames = enable_layers,
        .enabledExtensionCount = sizeof(exts) / sizeof(const char *),
        .ppEnabledExtensionNames = exts
    };

    ret = vkCreateInstance(&instance_create_info, NULL, &pdev->vk_instance);
    if (ret != VK_SUCCESS)
    {
        log_error("create vulkan instance failed: %d", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

#ifdef __VKLLM_DEBUG__
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |   // 错误
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | // 警告
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = NULL,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |  // 验证层消息/`debugPrintfEXT`输出
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | // 性能警告
                       VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,      // 与规范或性能无关的一般事件,
        .flags = 0,
    };

    PFN_vkCreateDebugUtilsMessengerEXT pfn =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(pdev->vk_instance, "vkCreateDebugUtilsMessengerEXT");
    if (pfn)
    {
        _CHECK_VK(pfn(pdev->vk_instance, &debugCreateInfo, NULL, &pdev->vk_debug_msgr));
    }

#endif

    return VKLLM_ERR_OK;
}

static vkllm_err_t init_physical_device(struct vkllm_context *context)
{
    uint32_t ndev = 0;
    struct vkllm_gpu_device *pdev = context->device;
    // FIXME: alloc dynamic
    VkPhysicalDevice physical_devices[VKLLM_MAX_PHY_DEVS] = {};
    VkResult ret = vkEnumeratePhysicalDevices(pdev->vk_instance, &ndev, NULL);

    if (ret != VK_SUCCESS)
    {
        log_error("vkEnumeratePhysicalDevices failed: %d", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    ret = vkEnumeratePhysicalDevices(pdev->vk_instance, &ndev, physical_devices);

    if (ndev <= pdev->vk_physical_dev.id)
    {
        log_error("target device id %u not found.", (unsigned int)pdev->vk_physical_dev.id);
        return VKLLM_ERR_DEV_NOT_FOUND;
    }

    pdev->vk_physical_dev.dev = physical_devices[pdev->vk_physical_dev.id];

    VkPhysicalDevice vk_physical_dev = pdev->vk_physical_dev.dev;

    vkGetPhysicalDeviceFeatures(vk_physical_dev, &pdev->vk_physical_dev.features);

    _CHECK_VK(
        vkEnumerateDeviceExtensionProperties(vk_physical_dev, NULL, &pdev->vk_physical_dev.n_ext_properties, NULL));

    _NEW_N_AND_CHECK(pdev->vk_physical_dev.ext_properties, VkExtensionProperties,
                     pdev->vk_physical_dev.n_ext_properties);

    _CHECK_VK(vkEnumerateDeviceExtensionProperties(vk_physical_dev, NULL, &pdev->vk_physical_dev.n_ext_properties,
                                                   pdev->vk_physical_dev.ext_properties));

    vkGetPhysicalDeviceMemoryProperties(vk_physical_dev, &pdev->vk_physical_dev.mem_properties);
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_dev, &pdev->vk_physical_dev.n_queue_family_properties, NULL);

    _NEW_N_AND_CHECK(pdev->vk_physical_dev.queue_family_properties, VkQueueFamilyProperties,
                     pdev->vk_physical_dev.n_queue_family_properties);

    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_dev, &pdev->vk_physical_dev.n_queue_family_properties,
                                             pdev->vk_physical_dev.queue_family_properties);

    vkGetPhysicalDeviceProperties(vk_physical_dev, &pdev->vk_physical_dev.properties);

    pdev->vk_physical_dev.feat_shader_fp16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    pdev->vk_physical_dev.feat_shader_fp16_int8.pNext = NULL;

    pdev->vk_physical_dev.feat_16bit_storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    pdev->vk_physical_dev.feat_16bit_storage.pNext = &pdev->vk_physical_dev.feat_shader_fp16_int8;

    pdev->vk_physical_dev.feat_8bit_storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    pdev->vk_physical_dev.feat_8bit_storage.pNext = &pdev->vk_physical_dev.feat_16bit_storage;
    pdev->vk_physical_dev.features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    pdev->vk_physical_dev.features2.pNext = &pdev->vk_physical_dev.feat_8bit_storage;
    vkGetPhysicalDeviceFeatures2(pdev->vk_physical_dev.dev, &pdev->vk_physical_dev.features2);

    pdev->vk_physical_dev.subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    pdev->vk_physical_dev.subgroup_properties.pNext = NULL;

    pdev->vk_physical_dev.properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    pdev->vk_physical_dev.properties2.pNext = &pdev->vk_physical_dev.subgroup_properties;

    vkGetPhysicalDeviceProperties2(pdev->vk_physical_dev.dev, &pdev->vk_physical_dev.properties2);

    uint32_t n_exts = pdev->vk_physical_dev.n_ext_properties;
    for (uint32_t i = 0; i < n_exts; ++i)
    {
        const char *ext_name = pdev->vk_physical_dev.ext_properties[i].extensionName;
        if (!strcmp(ext_name, VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME))
        {
            pdev->support_descriptor_templ_update = true;
        }
        else if (!strcmp(ext_name, VK_KHR_16BIT_STORAGE_EXTENSION_NAME))
        {
            pdev->support_16bit_storage = true;
        }
        else if (!strcmp(ext_name, VK_KHR_8BIT_STORAGE_EXTENSION_NAME))
        {
            pdev->support_8bit_storage = true;
        }
        else if (!strcmp(ext_name, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
        {
            pdev->support_fp16_arithmetic = true;
            pdev->support_int8_arithmetic = true;
        }
    }

    if (pdev->support_16bit_storage)
    {
        pdev->support_16bit_storage = pdev->vk_physical_dev.feat_16bit_storage.storageBuffer16BitAccess &&
                                      pdev->vk_physical_dev.feat_16bit_storage.storagePushConstant16;
    }

    if (pdev->support_8bit_storage)
    {
        pdev->support_8bit_storage = pdev->vk_physical_dev.feat_8bit_storage.storageBuffer8BitAccess &&
                                     pdev->vk_physical_dev.feat_8bit_storage.storagePushConstant8;
    }

    if (pdev->support_fp16_arithmetic)
    {
        pdev->support_fp16_arithmetic = pdev->vk_physical_dev.feat_shader_fp16_int8.shaderFloat16;
    }

    if (pdev->support_int8_arithmetic)
    {
        pdev->support_int8_arithmetic = pdev->vk_physical_dev.feat_shader_fp16_int8.shaderInt8;
    }

    return VKLLM_ERR_OK;
}

static vkllm_err_t init_logical_device(struct vkllm_context *context)
{
    struct vkllm_gpu_device *pdev = context->device;
    uint32_t n_queue = pdev->vk_physical_dev.n_queue_family_properties;
    VkDeviceQueueCreateInfo *dev_queue_create_infos = NULL;
    _NEW_N_AND_CHECK(dev_queue_create_infos, VkDeviceQueueCreateInfo, n_queue);

    uint32_t max_queue_counts = 0;
    for (uint32_t i = 0; i < n_queue; ++i)
    {
        if (pdev->vk_physical_dev.queue_family_properties[i].queueCount > max_queue_counts)
        {
            max_queue_counts = pdev->vk_physical_dev.queue_family_properties[i].queueCount;
        }
    }

    float *priorities = NULL;
    _NEW_N_AND_CHECK(priorities, float, max_queue_counts);

    for (uint32_t k = 0; k < max_queue_counts; ++k)
    {
        priorities[k] = .5f;
    }

    for (uint32_t i = 0; i < n_queue; ++i)
    {
        const uint32_t queue_counts = pdev->vk_physical_dev.queue_family_properties[i].queueCount;

        VkDeviceQueueCreateInfo queue_create_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                     .pNext = NULL,
                                                     .flags = 0,
                                                     .queueFamilyIndex = i,
                                                     .queueCount = queue_counts,
                                                     .pQueuePriorities = priorities};
        dev_queue_create_infos[i] = queue_create_info;
    }

    const char **dev_exts = NULL;
    _NEW_N_AND_CHECK(dev_exts, const char *, pdev->vk_physical_dev.n_ext_properties);
    uint32_t i_ext = 0;
#if __APPLE__
    dev_exts[i_ext++] = "VK_KHR_portability_subset";
#endif

    if (pdev->support_descriptor_templ_update)
    {
        dev_exts[i_ext++] = VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME;
    }

    if (pdev->support_16bit_storage)
    {
        dev_exts[i_ext++] = VK_KHR_16BIT_STORAGE_EXTENSION_NAME;
    }

    if (pdev->support_8bit_storage)
    {
        dev_exts[i_ext++] = VK_KHR_8BIT_STORAGE_EXTENSION_NAME;
    }

    if (pdev->support_fp16_arithmetic || pdev->support_int8_arithmetic)
    {
        dev_exts[i_ext++] = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
    }

    VkDeviceCreateInfo dev_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &pdev->vk_physical_dev.feat_8bit_storage,
        .flags = 0,
        .queueCreateInfoCount = n_queue,
        .pQueueCreateInfos = dev_queue_create_infos,
        .enabledExtensionCount = i_ext,
        .ppEnabledExtensionNames = dev_exts,
        .pEnabledFeatures = &pdev->vk_physical_dev.features,
    };

    VkResult ret = vkCreateDevice(pdev->vk_physical_dev.dev, &dev_create_info, NULL, &pdev->vk_dev);

    free(dev_queue_create_infos);
    free(priorities);
    free(dev_exts);

    if (ret == VK_SUCCESS)
    {
        return VKLLM_ERR_OK;
    }

    log_error("vkCreateDevice failed: %d", (int)ret);
    return VKLLM_ERR_VULKAN;
}

static vkllm_err_t init_vma_allocator(struct vkllm_context *context)
{
    struct vkllm_gpu_device *pdev = context->device;
    VmaAllocatorCreateInfo vma_create_info = {.flags = 0,
                                              .physicalDevice = pdev->vk_physical_dev.dev,
                                              .device = pdev->vk_dev,
                                              .preferredLargeHeapBlockSize = 0,
                                              .pAllocationCallbacks = NULL,
                                              .pDeviceMemoryCallbacks = NULL,
                                              .pHeapSizeLimit = NULL,
                                              .pVulkanFunctions = NULL,
                                              .instance = pdev->vk_instance,
                                              .vulkanApiVersion = pdev->api_version};

    VkResult ret = vmaCreateAllocator(&vma_create_info, &pdev->vma_allocator);

    if (ret == VK_SUCCESS)
    {
        return VKLLM_ERR_OK;
    }

    log_error("vmaCreateAllocator failed: %d", (int)ret);
    return VKLLM_ERR_VULKAN;
}

static vkllm_err_t init_gpu_device(struct vkllm_context *context)
{
    _CHECK(init_physical_device(context));
    _CHECK(init_logical_device(context));
    _CHECK(init_vma_allocator(context));
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_gpu_device_require_queue(struct vkllm_context *context, VkQueueFlagBits flags, uint32_t *type)
{
    struct vkllm_gpu_device *device = context->device;
    for (uint32_t i = 0; i < device->vk_physical_dev.n_queue_family_properties; ++i)
    {
        VkQueueFamilyProperties property = device->vk_physical_dev.queue_family_properties[i];
        if ((property.queueFlags & flags) == flags)
        {
            *type = i;
            return VKLLM_ERR_OK;
        }
    }

    return VKLLM_ERR_ARGS;
}

vkllm_err_t vkllm_gpu_device_new(struct vkllm_context *context, uint32_t id)
{
    _NEW_AND_CHECK(context->device, struct vkllm_gpu_device);

    struct vkllm_gpu_device *pdev = context->device;
    pdev->support_16bit_storage = false;
    pdev->support_8bit_storage = false;
    pdev->support_descriptor_templ_update = false;
    pdev->support_fp16_arithmetic = false;
    pdev->support_int8_arithmetic = false;
    pdev->support_pipeline_statistics = false;

    vkllm_err_t ret = create_instance(pdev);
    if (ret != VKLLM_ERR_OK)
    {
        goto err_create_instance;
    }

    pdev->vk_physical_dev.id = id;
    ret = init_gpu_device(context);
    if (ret != VKLLM_ERR_OK)
    {
        goto err_init_gpu_dev;
    }

#ifdef __VKLLM_DEBUG__
    log_info("gpu device info:");
    log_info("support_16bit_storage: %s", BOOL_S(pdev->support_16bit_storage));
    log_info("support_8bit_storage: %s", BOOL_S(pdev->support_8bit_storage));
    log_info("support_descriptor_templ_update: %s", BOOL_S(pdev->support_descriptor_templ_update));
    log_info("support_fp16_arithmetic: %s", BOOL_S(pdev->support_fp16_arithmetic));
    log_info("support_int8_arithmetic: %s", BOOL_S(pdev->support_int8_arithmetic));
    log_info("support_pipeline_statistics: %s", BOOL_S(pdev->support_pipeline_statistics));

    const uint32_t *max_group_counts = pdev->vk_physical_dev.properties.limits.maxComputeWorkGroupCount;
    log_info("group count limits.xyz = (%u, %u, %u)", max_group_counts[0], max_group_counts[1], max_group_counts[2]);
#endif
    return VKLLM_ERR_OK;

err_init_gpu_dev:
    vkDestroyInstance(pdev->vk_instance, NULL);
err_create_instance:
    free(pdev);
    return ret;
}

vkllm_err_t compute_group_counts(struct vkllm_context *context, uint32_t N, uint32_t local_x, uint32_t local_y,
                                 uint32_t local_z, uint32_t *group_x, uint32_t *group_y, uint32_t *group_z)
{
    _CHECK_ARGS(context && group_x && group_y && group_z);
    VkPhysicalDeviceLimits *limits = &context->device->vk_physical_dev.properties.limits;
    *group_x = 1;
    *group_y = 1;
    *group_z = 1;

    if (*group_x > limits->maxComputeWorkGroupCount[0])
    {
        *group_x = limits->maxComputeWorkGroupCount[0];
        N = N - *group_x * local_x;
        *group_y = (N + local_y - 1) / local_y;
        if (*group_y > limits->maxComputeWorkGroupCount[1])
        {
            *group_y = limits->maxComputeWorkGroupCount[1];
            N = N - *group_y * local_y;
            *group_z = (N + local_z - 1) / local_z;
        }
    }

    return VKLLM_ERR_OK;
}

void vkllm_gpu_device_free(struct vkllm_context *context)
{
    struct vkllm_gpu_device *pdev = context->device;
    vmaDestroyAllocator(pdev->vma_allocator);

    PFN_vkDestroyDebugUtilsMessengerEXT pfn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        pdev->vk_instance, "vkDestroyDebugUtilsMessengerEXT");
    if (pfn)
    {
        pfn(pdev->vk_instance, pdev->vk_debug_msgr, NULL);
    }
    vkDestroyDevice(pdev->vk_dev, NULL);
    vkDestroyInstance(pdev->vk_instance, NULL);
    free(pdev->vk_physical_dev.ext_properties);
    free(pdev->vk_physical_dev.queue_family_properties);
    free(pdev);
    context->device = NULL;
}
