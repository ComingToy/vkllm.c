load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

glsl_shader(
    # f32 in f32 load f32 out
    name = "vkllm_comp_shaders_f32_f32_f32",
    shaders = glob(["*.comp"]),
    hdrs = [
        "vkllm_shader_header.comp.h",
    ],
    extra_args = [
        "--target-env=vulkan1.2",
        "--target-spv=spv1.5",
        "-DIN_TYPE0=float",
        "-DIN_TYPE1=float",
        "-DIN_TYPE2=float",
        "-DIN_TYPE3=float",
        "-DLOAD_TYPE=float",
        "-DOUT_TYPE=float",
    ],
    tool = "//tools:binary_to_array",
    suffix = "_f32_f32_f32",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_comp_shaders_f16_f32_f32",
    shaders = glob(["*.comp"]),
    hdrs = [
        "vkllm_shader_header.comp.h",
    ],
    extra_args = [
        "--target-env=vulkan1.2",
        "--target-spv=spv1.5",
        "-D_SUPPORT_16BIT_STORAGE",
        "-DIN_TYPE0=float16_t",
        "-DIN_TYPE1=float16_t",
        "-DIN_TYPE2=float16_t",
        "-DIN_TYPE3=float16_t",
        "-DLOAD_TYPE=float",
        "-DOUT_TYPE=float",
    ],
    tool = "//tools:binary_to_array",
    suffix = "_f16_f32_f32",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_comp_shaders_f16_f16_f32",
    shaders = glob(["*.comp"]),
    hdrs = [
        "vkllm_shader_header.comp.h",
    ],
    extra_args = [
        "--target-env=vulkan1.2",
        "--target-spv=spv1.5",
        "-D_SUPPORT_16BIT_STORAGE",
        "-D_SUPPORT_FLOAT16_ARITHMETIC",
        "-DIN_TYPE0=float16_t",
        "-DIN_TYPE1=float16_t",
        "-DIN_TYPE2=float16_t",
        "-DIN_TYPE3=float16_t",
        "-DLOAD_TYPE=float",
        "-DOUT_TYPE=float",
    ],
    tool = "//tools:binary_to_array",
    suffix = "_f16_f16_f32",
    visibility = ["//visibility:public"]
)


cc_shader_library(
    name = 'vkllm_shaders',
    deps = [
        ':vkllm_comp_shaders_f16_f16_f32',
        ':vkllm_comp_shaders_f16_f32_f32',
        ':vkllm_comp_shaders_f32_f32_f32',
    ],
    visibility = ["//visibility:public"]
)
