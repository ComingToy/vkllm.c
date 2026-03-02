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
        name = "check",
        remote = "https://github.com/libcheck/check.git",
        commit = "11970a7e112dfe243a2e68773f014687df2900e8",
        build_file_content = _ALL_CONTENT,
    )
