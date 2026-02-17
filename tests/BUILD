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
    "//src:core",
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
