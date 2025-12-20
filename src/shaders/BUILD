load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

COMP_SHADER_SRCS = glob(["*.comp"])
COMP_SHADER_HDRS = glob(["*.comp.h"])

F32_F32_F32_ARGS = [

]

COMMON_ARGS = [
    "--target-env=vulkan1.2",
    "--target-spv=spv1.5",
]

glsl_shader(
    # f32 in f32 load f32 out
    name = "vkllm_comp_shaders_f32_f32_f32",
    shaders = COMP_SHADER_SRCS,
    hdrs = COMP_SHADER_HDRS,
    extra_args = [
        "-DIN_DTYPE=float",
        "-DLOAD_DTYPE=float",
        "-DOUT_DTYPE=float",
    ] + COMMON_ARGS,
    tool = "//tools:binary_to_array",
    suffix = "_f32_f32_f32",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_comp_shaders_f16_f32_f32",
    shaders = COMP_SHADER_SRCS,
    hdrs = COMP_SHADER_HDRS,
    extra_args = [
        "-D_SUPPORT_16BIT_STORAGE",
        "-DIN_DTYPE=float16_t",
        "-DLOAD_DTYPE=float",
        "-DOUT_DTYPE=float",
    ] + COMMON_ARGS,
    tool = "//tools:binary_to_array",
    suffix = "_f16_f32_f32",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_comp_shaders_f16_f32_f16",
    shaders = COMP_SHADER_SRCS,
    hdrs = COMP_SHADER_HDRS,
    extra_args = [
        "-D_SUPPORT_16BIT_STORAGE",
        "-DIN_DTYPE=float16_t",
        "-DLOAD_DTYPE=float",
        "-DOUT_DTYPE=float16_t",
    ] + COMMON_ARGS,
    tool = "//tools:binary_to_array",
    suffix = "_f16_f32_f16",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_comp_shaders_f16_f16_f32",
    shaders = COMP_SHADER_SRCS,
    hdrs = COMP_SHADER_HDRS,
    extra_args = [
        "-D_SUPPORT_16BIT_STORAGE",
        "-D_SUPPORT_FLOAT16_ARITHMETIC",
        "-DIN_DTYPE=float16_t",
        "-DLOAD_DTYPE=float16_t",
        "-DOUT_DTYPE=float",
    ] + COMMON_ARGS,
    tool = "//tools:binary_to_array",
    suffix = "_f16_f16_f32",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_comp_shaders_f16_f16_f16",
    shaders = COMP_SHADER_SRCS,
    hdrs = COMP_SHADER_HDRS,
    extra_args = [
        "-D_SUPPORT_16BIT_STORAGE",
        "-D_SUPPORT_FLOAT16_ARITHMETIC",
        "-DIN_DTYPE=float16_t",
        "-DLOAD_DTYPE=float16_t",
        "-DOUT_DTYPE=float16_t",
    ] + COMMON_ARGS,
    tool = "//tools:binary_to_array",
    suffix = "_f16_f16_f16",
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllm_shaders',
    deps = [
        ':vkllm_comp_shaders_f16_f16_f16',
        ':vkllm_comp_shaders_f16_f16_f32',
        ':vkllm_comp_shaders_f16_f32_f32',
        ':vkllm_comp_shaders_f32_f32_f32',
    ],
    visibility = ["//visibility:public"]
)
