cc_library(
    name = "core",
    srcs = [
        "vkllm_gpu_device.c",
        "vkllm_context.c",
    ],
    hdrs = [
        "vkllm_gpu_device.h",
        "vkllm_context.h",
        "vkllm_errors.h",
        "vkllm_errors.inc",
        "vkllm_common.h",
    ],
    deps = [
		"//vulkan_rules:vulkan_cc_library",
        "@zlog//:zlog",
    ],
    copts = [
        "-std=c11"
    ],
    visibility = ["//visibility:public"],
)
