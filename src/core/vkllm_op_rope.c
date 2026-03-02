#include "vkllm_op_rope.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

static vkllm_err_t vkllm_op_rope_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                              struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline && tensor->srcs[0]);
    *pipeline = NULL;

    // RoPE operates in-place on the input tensor
    struct vkllm_tensor *in0 = tensor->srcs[0];
    struct vkllm_op_rope_params *params = (struct vkllm_op_rope_params *)tensor->params;
    int neox_style = (int)params->neox_style;

    if (in0->dtype == vkllm_dtype_float16)
    {
        if (context->device->support_fp16_arithmetic)
        {
            *pipeline = context->pipelines.rope.f16f16[neox_style];
        }
        else
        {
            *pipeline = context->pipelines.rope.f16f32[neox_style];
        }
        return VKLLM_ERR_OK;
    }
    else if (in0->dtype == vkllm_dtype_float32)
    {
        *pipeline = context->pipelines.rope.f32f32[neox_style];
        return VKLLM_ERR_OK;
    }
    else
    {
        log_error("unsupported op input dtype: %s", vkllm_dtype_s(in0->dtype));
        return VKLLM_ERR_ARGS;
    }
}

vkllm_err_t vkllm_op_rope_init(struct vkllm_context *context, struct vkllm_commands *commands,
                               struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_ROPE);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    _CHECK_ARGS(in0);

    // Validate that shapes match (RoPE operates in-place or produces same-shaped output)
    _CHECK_ARGS(in0->shapes[0] == tensor->shapes[0]);
    _CHECK_ARGS(in0->shapes[1] == tensor->shapes[1]);
    _CHECK_ARGS(in0->shapes[2] == tensor->shapes[2]);
    _CHECK_ARGS(in0->shapes[3] == tensor->shapes[3]);

    struct vkllm_pipeline *pipeline = NULL;
    _CHECK(vkllm_op_rope_get_pipeline(context, tensor, &pipeline));
    tensor->pipeline = pipeline;

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_rope_run(struct vkllm_context *context, struct vkllm_commands *commands,
                              struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_ROPE);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    _CHECK_ARGS(in0);

    // Extract RoPE parameters from tensor params
    struct vkllm_op_rope_params *params = (struct vkllm_op_rope_params *)tensor->params;
    uint32_t offset = params ? params->offset : 0;
    float base = params ? params->base : 10000.0f;

    struct vkllm_dtype_info dtype_info;
    uint32_t out_strides[4], in_strides[4];
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));
    _DIV4_S(in0->strides, dtype_info.bytes, in_strides);
    _DIV4_S(tensor->strides, dtype_info.bytes, out_strides);

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 128));
    vkllm_shader_constants_append_n(constants, in0->shapes, 4);
    vkllm_shader_constants_append_n(constants, in_strides, 4);
    vkllm_shader_constants_append_n(constants, tensor->shapes, 4);
    vkllm_shader_constants_append_n(constants, out_strides, 4);
    vkllm_shader_constants_append(constants, offset);
    if (tensor->dtype == vkllm_dtype_float16)
    {
        vkllm_shader_constants_append(constants, base);
    }
    else
    {
        vkllm_shader_constants_append(constants, base);
    }

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 2), err, free_constants_out);

    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, tensor);

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    // Calculate work group dimensions
    // RoPE processes pairs of elements along the last dimension
    // Total threads = B * C * H * (W/2)
    uint32_t B = tensor->shapes[0];
    uint32_t C = tensor->shapes[1];
    uint32_t H = tensor->shapes[2];
    uint32_t W = tensor->shapes[3] / 2; // Process pairs
    uint32_t N = B * C * H * W;

    uint32_t group_x = (N + pipeline->shader_info.local_x - 1) / pipeline->shader_info.local_x;
    uint32_t group_y = 1, group_z = 1;

    VkPhysicalDeviceLimits *limits = &context->device->vk_physical_dev.properties.limits;
    if (group_x > limits->maxComputeWorkGroupCount[0])
    {
        log_info("group_x %u > %u, adjust to max\n", group_x, limits->maxComputeWorkGroupCount[0]);
        group_x = limits->maxComputeWorkGroupCount[0];
        N = N - group_x * pipeline->shader_info.local_x;
        group_y = (N + pipeline->shader_info.local_y - 1) / pipeline->shader_info.local_y;
        if (group_y > limits->maxComputeWorkGroupCount[1])
        {
            log_info("group_y %u > %u, adjust to max\n", group_y, limits->maxComputeWorkGroupCount[1]);
            group_y = limits->maxComputeWorkGroupCount[1];
            N = N - group_y * pipeline->shader_info.local_y;
            group_z = (N + pipeline->shader_info.local_z - 1) / pipeline->shader_info.local_z;
        }
    }

    _CHECK_JUMP(
        vkllm_commands_pipeline(context, commands, pipeline, bindings, NULL, constants, group_x, group_y, group_z), err,
        free_bindings_out);

free_bindings_out:
    vkllm_array_ptr_free(bindings);
free_constants_out:
    vkllm_shader_constants_free(constants);
    return err;
}

vkllm_err_t vkllm_op_rope_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                   struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);
    return VKLLM_ERR_OK;
}
