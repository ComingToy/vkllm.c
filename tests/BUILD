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


COPTS = [
    "-std=c11",
]

cc_test(
    name = "test_transfer",
    srcs = ["test_transfer.c"],
    deps = [
        "//src:core",
        "//thirdparty/check:check",
    ],
    copts = COPTS,
    linkopts = LINK_OPTS,
)

cc_test(
    name = "test_op_add",
    srcs = ["test_op_add.c"],
    deps = [
        "//src:core",
        "//thirdparty/check:check",
    ],
    copts = COPTS,
    linkopts = LINK_OPTS,
)
