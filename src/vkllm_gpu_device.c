#include "vkllm_gpu_device.h"

#include <stdlib.h>

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

static vkllm_err_t init_gpu_device(struct vkllm_context* context,
                                   struct vkllm_gpu_device* dev) {
    uint32_t ndev = 0;
}

vkllm_err_t new_gpu_device(struct vkllm_context* context, int id,
                           struct vkllm_gpu_device** ppdev) {
    _NEW_AND_CHECK(*ppdev, struct vkllm_gpu_device);

    struct vkllm_gpu_device* pdev = *ppdev;
    pdev->id = id;

    vkllm_err_t ret = create_instance(context, pdev);
    if (ret != VKLLM_ERR_OK) {
        goto err_create_instance;
    }

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
