load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "gguf-tools",
        remote = "https://github.com/antirez/gguf-tools.git",
        commit = 'a3257ff3cb8aed8b60ba3243c70b85a17491d7d6',
        build_file = "//thirdparty/gguf-tools:gguf-tools.BUILD"
    )
