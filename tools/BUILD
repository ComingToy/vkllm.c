load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name='binary_to_array',
    srcs=['binary_to_array.py'],
    visibility = ["//visibility:public"],
    deps = select({
        '@platforms//os:macos': ["@pypi_macos//jinja2"],
        '@platforms//os:linux': ["@pypi//jinja2"]
    })
)
