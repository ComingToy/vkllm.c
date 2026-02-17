load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

COMP_SHADER_SRCS = glob(["*.comp"])
COMP_SHADER_HDRS = glob(["*.comp.h"])

COMMON_ARGS = [
    "--target-env=vulkan1.2",
    "--target-spv=spv1.5",
]

glsl_shader(
    name = "vkllm_add_shaders",
    shaders = [
        "vkllm_add.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-DIN_DTYPE=float -DLOAD_DTYPE=float -DOUT_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -DIN_DTYPE=float16_t -DLOAD_DTYPE=float -DOUT_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DIN_DTYPE=float16_t -DLOAD_DTYPE=float16_t -DOUT_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DIN_DTYPE=float16_t -DLOAD_DTYPE=float16_t -DOUT_DTYPE=float16_t",
    ],
    suffixes = [
        "_f32f32f32",
        "_f16f32f32",
        "_f16f16f32",
        "_f16f16f16",
    ],
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_embedding_shaders",
    shaders = [
        "vkllm_embedding.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-D_SUPPORT_16BIT_STORAGE -DDTYPE=float16_t",
        "-DDTYPE=float",
    ],
    suffixes = [
        "_f16",
        "_f32",
    ],
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "vkllm_rmsnorm_shaders",
    shaders = [
        "vkllm_rmsnorm.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-D_SUPPORT_16BIT_STORAGE -DIN0_DTYPE=float16_t -DIN1_DTYPE=float16_t -DLOAD_DTYPE=float -DOUT_DTYPE=float",
        "-DIN0_DTYPE=float -DIN1_DTYPE=float -DLOAD_DTYPE=float -DOUT_DTYPE=float",
    ],
    suffixes = [
        "_f16f32f32",
        "_f32f32f32",
    ],
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllm_shaders',
    deps = [
        ':vkllm_add_shaders',
        ':vkllm_embedding_shaders',
        ':vkllm_rmsnorm_shaders',
    ],
    visibility = ["//visibility:public"]
)
