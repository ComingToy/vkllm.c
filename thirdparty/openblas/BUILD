load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "openblas",
    cache_entries = {
        "CMAKE_C_FLAGS": "-fPIC",
    },
    build_args = [
        "-j6",
    ],
    lib_source = "@openblas//:all_srcs",
    out_static_libs = ["libopenblas.a"],
    visibility = ["//visibility:public"],
)

