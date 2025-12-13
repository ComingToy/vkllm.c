cc_library(
    name = "core",
    srcs = [
        "vkllm_gpu_device.c",
        "vkllm_context.c",
        "vkllm_errors.c",
        "vkllm_dtypes.c",
        "vkllm_tensor.c",
        "vkllm_commands.c",
        "vkllm_pipeline.c",
        "vkllm_ops.c",
        "vkllm_op_vector_add.c",
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
        "vkllm_commands.h",
        "vkllm_pipeline.h",
        "vkllm_array.h",
        "vkllm_ops.h",
        "vkllm_op_vector_add.h",
    ],
    deps = [
		"//vulkan_rules:vulkan_cc_library",
        "@log.c//:log.c",
		":vma_impl",
        "//src/shaders:vkllm_shaders",
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
		"-std=c++17",
	],
	deps = [
		"@vma//:vma",
		"//vulkan_rules:vulkan_cc_library",
	],
	visibility = [
		"//visibility:public",
	],
)
