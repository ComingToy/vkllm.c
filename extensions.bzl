load('//vulkan_rules:linux_lib.bzl', 'vulkan_linux_cc_lib')
load('//vulkan_rules:macos_lib.bzl', 'vulkan_macos_cc_lib')
load('//thirdparty/log.c:repo.bzl', logc_repo = "repo")
load('//thirdparty/check:repo.bzl', check_repo = "repo")
load('//thirdparty/VulkanMemoryAllocator:repo.bzl', vma_repo = "repo")


def _vkllm_vulkan_deps_impl(module_ctx):
    vulkan_linux_cc_lib(name='vulkan_linux')
    vulkan_macos_cc_lib(name='vulkan_macos')

def _vkllm_thirdparty_deps_impl(module_ctx):
    logc_repo()
    check_repo()
    vma_repo()

vkllm_vulkan_deps = module_extension(
    implementation =  _vkllm_vulkan_deps_impl,
) 

vkllm_thirdparty_deps = module_extension(
    implementation = _vkllm_thirdparty_deps_impl,
)
