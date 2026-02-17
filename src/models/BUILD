cc_library(
    name = "vkllm_llama2_models",
    srcs = [
        "vkllm_llama2_layers.c",
    ],
    hdrs = [
        "vkllm_llama2_layers.h",
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
