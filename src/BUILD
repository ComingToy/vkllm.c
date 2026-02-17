cc_library(
    name = "core",
    srcs = [
        "vkllm_gpu_device.c",
        "vkllm_context.c",
        "vkllm_errors.c",
        "vkllm_dtypes.c",
    ],
    hdrs = [
        "vkllm_gpu_device.h",
        "vkllm_context.h",
        "vkllm_errors.h",
		"vkllm_errors.c",
        "vkllm_errors.inc",
        "vkllm_dtypes.inc",
        "vkllm_dtypes.h",
        "vkllm_common.h",
        "vkllm_tensor.h",
        "vkllm_ops.h",
    ],
    deps = [
		"//vulkan_rules:vulkan_cc_library",
        "@log.c//:log.c",
		":vma_impl",
    ],
    copts = [
        "-std=c11"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
	name = "vma_impl",
	srcs = [
		"vulkan_mem_allocator_imp.cpp",
	],
	copts = [
		"-std=c++11",
	],
	deps = [
		"@vma//:vma",
	],
	visibility = [
		"//visibility:public",
	],
)
