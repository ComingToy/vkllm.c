load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "check",
    cache_entries = {
        "CMAKE_C_FLAGS": "-fPIC",
    },
    lib_source = "@check//:all_srcs",
    out_static_libs = ["libcheck.a"],
    visibility = ["//visibility:public"],
)

