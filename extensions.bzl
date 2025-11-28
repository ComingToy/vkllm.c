load('//vulkan_rules:linux_lib.bzl', 'vulkan_linux_cc_lib')
load('//vulkan_rules:macos_lib.bzl', 'vulkan_macos_cc_lib')

def _vkllm_vulkan_deps_impl(module_ctx):
    vulkan_linux_cc_lib(name='vulkan_linux')
    vulkan_macos_cc_lib(name='vulkan_macos')
    # native.register_toolchains('//vulkan_rules:glsl_linux_toolchain')
    # native.register_toolchains('//vulkan_rules:glsl_macos_toolchain')

vkllm_vulkan_deps = module_extension(
    implementation =  _vkllm_vulkan_deps_impl,
) 
