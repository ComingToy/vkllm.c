load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

def repo():
    git_repository(
        name = "openblas",
        remote = "https://github.com/OpenMathLib/OpenBLAS.git",
        commit = "993fad6aebbce34a97d3f8c34d6d79d35b64cc48",
        build_file_content = _ALL_CONTENT,
    )
