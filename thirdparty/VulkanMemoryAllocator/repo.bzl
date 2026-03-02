load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
def repo():
    git_repository(
        name = "vma",
        remote = "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git",
        tag = 'v3.3.0',
        build_file = "//thirdparty/VulkanMemoryAllocator:VulkanMemoryAllocator.BUILD"
    )
