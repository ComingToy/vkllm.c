LINUX_LINK_OPTS = [
    "-lrt",
    "-lpthread",
]

MACOS_LINK_OPTS = [
    "-lpthread",
]

LINK_OPTS = select({
    "//:macos": MACOS_LINK_OPTS,
    "//:linux": LINUX_LINK_OPTS,
    "//conditions:default": [], 
})

cc_library(
    name = "vkllm_test_common",
    srcs = [],
    hdrs = [
        "vkllm_test_common.h",
    ],
)

DEPS = [
    "//src:vkllm",
    "//thirdparty/check:check",
    ":vkllm_test_common",
]


COPTS = [
    "-std=c11",
]

cc_test(
    name = "vkllm_test_transfer",
    srcs = ["vkllm_test_transfer.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_add",
    srcs = ["vkllm_test_op_add.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_embedding",
    srcs = ["vkllm_test_op_embedding.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_rmsnorm",
    srcs = ["vkllm_test_op_rmsnorm.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_matmul",
    srcs = ["vkllm_test_op_matmul.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_permute",
    srcs = ["vkllm_test_op_permute.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_rope",
    srcs = ["vkllm_test_op_rope.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_arg_max",
    srcs = ["vkllm_test_op_arg_max.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_softmax",
    srcs = ["vkllm_test_op_softmax.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_bin",
    srcs = ["vkllm_test_op_bin.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_copy",
    srcs = ["vkllm_test_op_copy.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_hashset",
    srcs = ["vkllm_test_hashset.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_hashmap",
    srcs = ["vkllm_test_hashmap.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_hashset_perf",
    srcs = ["vkllm_test_hashset_perf.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_tensor_reshape",
    srcs = ["vkllm_test_tensor_reshape.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_ffn_up_and_gate",
    srcs = ["vkllm_test_op_ffn_up_and_gate.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "vkllm_test_op_update_rows",
    srcs = ["vkllm_test_op_update_rows.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)


cc_test(
    name = "vkllm_test_infer_gguf",
    srcs = ["vkllm_test_infer_gguf.c"],
    deps = DEPS,
    copts = COPTS,
    linkopts = LINK_OPTS,
)
