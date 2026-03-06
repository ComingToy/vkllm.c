load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

COMP_SHADER_SRCS = glob(["*.comp"])
COMP_SHADER_HDRS = glob(["*.comp.h"])

COMMON_ARGS = [
    "--target-env=vulkan1.2",
    "--target-spv=spv1.5",
]

glsl_shader(
    name = "vkllm_bin_op_shaders",
    shaders = [
        "vkllm_bin_op.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-DIN_DTYPE=float -DLOAD_DTYPE=float -DOUT_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -DIN_DTYPE=float16_t -DLOAD_DTYPE=float -DOUT_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DIN_DTYPE=float16_t -DLOAD_DTYPE=float16_t -DOUT_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DIN_DTYPE=float16_t -DLOAD_DTYPE=float16_t -DOUT_DTYPE=float16_t",
        "-D_SUPPORT_16BIT_STORAGE -DIN_DTYPE=float16_t -DLOAD_DTYPE=float -DOUT_DTYPE=float16_t",
    ],
    suffixes = [
        "_f32f32f32",
        "_f16f32f32",
        "_f16f16f32",
        "_f16f16f16",
        "_f16f32f16",
    ],
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
        "-D_SUPPORT_16BIT_STORAGE -DIN0_DTYPE=float16_t -DIN1_DTYPE=float -DLOAD_DTYPE=float -DOUT_DTYPE=float16_t",
    ],
    suffixes = [
        "_f16f32f16",
    ],
)

glsl_shader(
    name = "vkllm_matmul_shaders",
    shaders = [
        "vkllm_matmul_t0.comp",
        "vkllm_matmul_t1.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-DIN0_DTYPE=float -DIN1_DTYPE=float -DSMEM_VEC2=vec2 -DOUT_DTYPE=float -DACC_DTYPE=float",
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DIN0_DTYPE=float16_t -DIN1_DTYPE=float16_t -DSMEM_VEC2=f16vec2 -DACC_DTYPE=float -DOUT_DTYPE=float16_t -DBROADCASE_A=0 -DBROADCASE_B=0",
        "-D_SUPPORT_16BIT_STORAGE -DIN0_DTYPE=float16_t -DIN1_DTYPE=float16_t -DSMEM_VEC2=vec2 -DLOAD_DTYPE=float -DACC_DTYPE=float -DOUT_DTYPE=float16_t -DBROADCASE_A=0 -DBROADCASE_B=0",
    ],
    suffixes = [
        "f32f32f32",
        "f16f16f16",
        "f16f32f16",
    ],
)

glsl_shader(
    name = "vkllm_rope_shaders",
    shaders = [
        "vkllm_rope.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DDTYPE=float16_t -DLOAD_DTYPE=float16_t",
        "-D_SUPPORT_16BIT_STORAGE -DDTYPE=float16_t -DLOAD_DTYPE=float",
        "-DDTYPE=float -DLOAD_DTYPE=float",
    ],
    suffixes = [
        "_f16f16",
        "_f16f32",
        "_f32f32",
    ],
)

glsl_shader(
    name = "vkllm_softmax_shaders",
    shaders = [
        "vkllm_softmax.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-D_SUPPORT_16BIT_STORAGE -D_SUPPORT_FLOAT16_ARITHMETIC -DDTYPE=float16_t -DLOAD_DTYPE=float16_t",
        "-D_SUPPORT_16BIT_STORAGE -DDTYPE=float16_t -DLOAD_DTYPE=float",
        "-DDTYPE=float -DLOAD_DTYPE=float",
    ],
    suffixes = [
        "_f16f16",
        "_f16f32",
        "_f32f32",
    ],
)

glsl_shader(
    name = "vkllm_copy_shaders",
    shaders = [
        "vkllm_copy.comp",
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
)

glsl_shader(
    name = "vkllm_ffn_shaders",
    shaders = [
        "vkllm_ffn_up_and_gate.comp",
    ],
    hdrs = COMP_SHADER_HDRS,
    extra_args = COMMON_ARGS,
    tool = "//tools:binary_to_array",
    macros = [
        "-D_SUPPORT_16BIT_STORAGE -DIN_DTYPE=float16_t -DLOAD_DTYPE=float -DOUT_DTYPE=float16_t",
        "-D_SUPPORT_16BIT_STORAGE -DIN_DTYPE=float16_t -DLOAD_DTYPE=float -DOUT_DTYPE=float",
        "-DIN_DTYPE=float -DLOAD_DTYPE=float -DOUT_DTYPE=float",
    ],
    suffixes = [
        "_f16f32f16",
        "_f16f32f32",
        "_f32f32f32",
    ],
)

glsl_shader(
    name = "vkllm_update_rows",
    shaders = [
        "vkllm_update_rows.comp",
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
)


cc_shader_library(
    name = "vkllm_shaders",
    deps = [
        ":vkllm_bin_op_shaders",
        ":vkllm_embedding_shaders",
        ":vkllm_rmsnorm_shaders",
        ":vkllm_matmul_shaders",
        ":vkllm_rope_shaders",
        ":vkllm_softmax_shaders",
        ":vkllm_copy_shaders",
        ":vkllm_ffn_shaders",
        ":vkllm_update_rows",
    ],
    visibility = ["//visibility:public"]
)
