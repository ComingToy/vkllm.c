cc_library(
    name = "vkllm_llama2_models",
    srcs = [
        "vkllm_llama2_layers.c",
        "vkllm_models_llama2.c",
    ],
    hdrs = [
        "vkllm_llama2_layers.h",
        "vkllm_models_llama2.h",
    ],
    deps = [
        "//src/core:core",
        "@gguf-tools//:gguf",
    ],
    copts = [
        "-std=c11"
    ],

    linkstatic = True,
    visibility = ["//visibility:public"],
)
