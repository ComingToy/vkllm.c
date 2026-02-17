load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

glsl_shader(
    name = "vkllm_comp_shaders",
    shaders = glob(["*.comp"]),
    hdrs = [
        "vkllm_shader_header.comp.h",
    ],
    extra_args = [
        "--target-env=vulkan1.2",
        "--target-spv=spv1.5",
    ],
    tool = "//tools:binary_to_array",
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllm_shaders',
    deps = [':vkllm_comp_shaders'],
    visibility = ["//visibility:public"]
)
