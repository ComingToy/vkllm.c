#include "vkllm_gpu_device.h"

#include <stdlib.h>
#include <string.h>

#include "vkllm_common.h"
#include "vulkan/vulkan_core.h"

static int create_instance(struct vkllm_context* context,
                           struct vkllm_gpu_device* pdev) {
    VkResult ret = vkEnumerateInstanceVersion(&pdev->api_version);
    if (ret != VK_SUCCESS) {
        zlog_error(context->zlog_c,
                   "call vkEnumerateInstanceVersion failed: %d", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = "vkllm.c",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = NULL,
        .engineVersion = 0,
        .apiVersion = pdev->api_version};

    const char* enable_layers[] = {
#ifdef __VKLLM_DEBUG__
        "VK_LAYER_KHRONOS_validation",
#endif
    };

    const char* exts[] = {
#ifdef __APPLE__
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
    };

    VkInstanceCreateInfo instance_create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = VK_KHR_portability_enumeration,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = sizeof(enable_layers) / sizeof(const char*),
        .ppEnabledLayerNames = enable_layers,
        .enabledExtensionCount = sizeof(exts) / sizeof(const char*),
        .ppEnabledExtensionNames = exts};

    ret = vkCreateInstance(&instance_create_info, NULL, &pdev->instance);
    if (ret != VK_SUCCESS) {
        zlog_error(context->zlog_c, "create vulkan instance failed: %d",
                   (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    return VKLLM_ERR_OK;
}

static vkllm_err_t init_physical_device(struct vkllm_context* context,
                                        struct vkllm_gpu_device* pdev) {
    uint32_t ndev = 0;
    // FIXME: alloc dynamic
    VkPhysicalDevice physical_devices[VKLLM_MAX_PHY_DEVS] = {};
    VkResult ret =
        vkEnumeratePhysicalDevices(pdev->instance, &ndev, physical_devices);

    if (ret != VK_SUCCESS) {
        zlog_error(context->zlog_c, "vkEnumeratePhysicalDevices failed: %d",
                   (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    if (ndev <= pdev->vk_physical_dev.id) {
        zlog_error(context->zlog_c, "target device id %u not found.",
                   (unsigned int)pdev->vk_physical_dev.id);
        return VKLLM_ERR_DEV_NOT_FOUND;
    }

    pdev->vk_physical_dev.dev = physical_devices[pdev->vk_physical_dev.id];

    VkPhysicalDevice vk_physical_dev = pdev->vk_physical_dev.dev;

    vkGetPhysicalDeviceFeatures(vk_physical_dev,
                                &pdev->vk_physical_dev.features);

    ret = vkEnumerateDeviceExtensionProperties(
        vk_physical_dev, NULL, &pdev->vk_physical_dev.n_ext_properties, NULL);

    if (ret != VK_SUCCESS) {
        zlog_error(context->zlog_c,
                   "vkEnumerateDeviceExtensionProperties failed: %d", (int)ret);
        return VKLLM_ERR_VULKAN;
    }

    _NEW_N_AND_CHECK(pdev->vk_physical_dev.ext_properties,
                     VkExtensionProperties,
                     pdev->vk_physical_dev.n_ext_properties);

    vkGetPhysicalDeviceMemoryProperties(vk_physical_dev,
                                        &pdev->vk_physical_dev.mem_properties);
    vkGetPhysicalDeviceQueueFamilyProperties(
        vk_physical_dev, &pdev->vk_physical_dev.n_queue_family_properties,
        NULL);

    _NEW_N_AND_CHECK(pdev->vk_physical_dev.queue_family_properties,
                     VkQueueFamilyProperties,
                     pdev->vk_physical_dev.n_queue_family_properties);

    vkGetPhysicalDeviceQueueFamilyProperties(
        vk_physical_dev, &pdev->vk_physical_dev.n_queue_family_properties,
        pdev->vk_physical_dev.queue_family_properties);

    vkGetPhysicalDeviceProperties(vk_physical_dev,
                                  &pdev->vk_physical_dev.properties);

    pdev->vk_physical_dev.subgroup_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    pdev->vk_physical_dev.subgroup_properties.pNext = NULL;

    pdev->vk_physical_dev.properties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    pdev->vk_physical_dev.properties2.pNext =
        &pdev->vk_physical_dev.subgroup_properties;

    vkGetPhysicalDeviceProperties2(pdev->vk_physical_dev.dev,
                                   &pdev->vk_physical_dev.properties2);

    zlog_info(context->zlog_c, "physical device subgroup size = %u",
              pdev->vk_physical_dev.subgroup_properties.subgroupSize);
    return VKLLM_ERR_OK;
}

static vkllm_err_t init_logical_device(struct vkllm_context* context,
                                       struct vkllm_gpu_device* pdev) {
    uint32_t n_queue = pdev->vk_physical_dev.n_queue_family_properties;
    VkDeviceQueueCreateInfo* dev_queue_create_infos = NULL;
    _NEW_N_AND_CHECK(dev_queue_create_infos, VkDeviceQueueCreateInfo, n_queue);

    uint32_t max_queue_counts = 0;
    for (uint32_t i = 0; i < n_queue; ++i) {
        if (pdev->vk_physical_dev.queue_family_properties[i].queueCount >
            max_queue_counts) {
            max_queue_counts =
                pdev->vk_physical_dev.queue_family_properties[i].queueCount;
        }
    }

    float* priorities = NULL;
    _NEW_N_AND_CHECK(priorities, float, max_queue_counts);

    for (uint32_t k = 0; k < max_queue_counts; ++k) {
        priorities[k] = .5f;
    }

    for (uint32_t i = 0; i < n_queue; ++i) {
        const uint32_t queue_counts =
            pdev->vk_physical_dev.queue_family_properties[i].queueCount;

        VkDeviceQueueCreateInfo queue_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queueFamilyIndex = i,
            .queueCount = queue_counts,
            .pQueuePriorities = priorities};
        dev_queue_create_infos[i] = queue_create_info;
    }

    uint32_t n_exts = pdev->vk_physical_dev.n_ext_properties;
    const char** dev_exts = NULL;
    _NEW_N_AND_CHECK(dev_exts, const char*, n_exts);
    for (uint32_t i = 0, k = 0; i < n_exts; ++i) {
        const char* ext_name =
            pdev->vk_physical_dev.ext_properties->extensionName;
        if (!strcmp(ext_name,
                    VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME)) {
            dev_exts[k++] = ext_name;
            pdev->support_descriptor_templ_update = true;
        } else if (!strcmp(ext_name, VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) {
            dev_exts[k++] = ext_name;
            pdev->support_16bit_storage = true;
        } else if (!strcmp(ext_name, VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) {
            dev_exts[k++] = ext_name;
            pdev->support_8bit_storage = true;
        } else if (!strcmp(ext_name,
                           VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) {
            dev_exts[k++] = ext_name;
            pdev->support_fp16_arithmetic = true;
            pdev->support_int8_arithmetic = true;
        }
    }

    VkPhysicalDeviceShaderFloat16Int8Features feat_fp16_int8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .pNext = NULL};

    VkDeviceCreateInfo dev_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueCreateInfoCount = n_queue,
        .pQueueCreateInfos = dev_queue_create_infos,
    };
}

static vkllm_err_t init_gpu_device(struct vkllm_context* context,
                                   struct vkllm_gpu_device* pdev) {
    _CHECK(init_physical_device(context, pdev));
    _CHECK(init_logical_device(context, pdev));

    return VKLLM_ERR_OK;
}

vkllm_err_t new_gpu_device(struct vkllm_context* context, uint32_t id,
                           struct vkllm_gpu_device** ppdev) {
    _NEW_AND_CHECK(*ppdev, struct vkllm_gpu_device);

    struct vkllm_gpu_device* pdev = *ppdev;

    vkllm_err_t ret = create_instance(context, pdev);
    if (ret != VKLLM_ERR_OK) {
        goto err_create_instance;
    }

    pdev->vk_physical_dev.id = id;
    ret = init_gpu_device(context, pdev);
    if (ret != VKLLM_ERR_OK) {
        goto err_init_gpu_dev;
    }

err_init_gpu_dev:
    vkDestroyInstance(pdev->instance, NULL);
err_create_instance:
    free(pdev);
    return ret;
}
