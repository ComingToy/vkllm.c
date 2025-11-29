load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "zlog",
        remote = "https://github.com/HardySimpson/zlog.git",
        tag = "1.2.18",
        build_file = "//thirdparty/zlog:zlog.BUILD"
    )
