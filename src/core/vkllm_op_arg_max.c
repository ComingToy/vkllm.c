#include "vkllm_op_arg_max.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

static vkllm_err_t vkllm_op_arg_max_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                                 struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline && tensor->srcs[0]);
    *pipeline = NULL;

    if (tensor->dtype != vkllm_dtype_uint32)
    {
        log_error("arg_max input and output dtype mismatch: input=%s, output=%s", vkllm_dtype_s(tensor->srcs[0]->dtype),
                  vkllm_dtype_s(tensor->dtype));
        return VKLLM_ERR_ARGS;
    }

    if (tensor->srcs[0]->dtype == vkllm_dtype_float16)
    {
        if (context->device->support_fp16_arithmetic)
        {
            *pipeline = context->pipelines.arg_max.f16f16;
        }
        else
        {
            *pipeline = context->pipelines.arg_max.f16f32;
        }
        return VKLLM_ERR_OK;
    }
    else if (tensor->srcs[0]->dtype == vkllm_dtype_float32)
    {
        *pipeline = context->pipelines.arg_max.f32f32;
        return VKLLM_ERR_OK;
    }
    else
    {
        log_error("unsupported arg_max dtype: %s", vkllm_dtype_s(tensor->srcs[0]->dtype));
        return VKLLM_ERR_ARGS;
    }
}

vkllm_err_t vkllm_op_arg_max_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                  struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_ARG_MAX);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    _CHECK_ARGS(in0);

    _CHECK_ARGS(in0->shapes[0] == tensor->shapes[0]);
    _CHECK_ARGS(in0->shapes[1] == tensor->shapes[1]);
    _CHECK_ARGS(in0->shapes[2] == tensor->shapes[2]);
    _CHECK_ARGS(tensor->shapes[3] == 1);

    struct vkllm_pipeline *pipeline = NULL;
    _CHECK(vkllm_op_arg_max_get_pipeline(context, tensor, &pipeline));
    tensor->pipeline = pipeline;
    _CHECK(vkllm_pipeline_alloc_desc_set(context, tensor->pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_arg_max_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                 struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_ARG_MAX);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    _CHECK_ARGS(in0);

    struct vkllm_tensor *out0 = tensor;

    struct vkllm_dtype_info dtype_info;
    uint32_t in_strides[4], out0_strides[4];
    _CHECK(vkllm_get_dtype_info(in0->dtype, &dtype_info));
    _DIV4_S(in0->strides, dtype_info.bytes, in_strides);
    _DIV4_S(out0->strides, dtype_info.bytes, out0_strides);

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 96));
    vkllm_shader_constants_append_n(constants, in0->shapes, 4);
    vkllm_shader_constants_append_n(constants, in_strides, 4);
    vkllm_shader_constants_append_n(constants, out0->shapes, 4);
    vkllm_shader_constants_append_n(constants, out0_strides, 4);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, free_constants_out);

    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, out0);

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    uint32_t B = in0->shapes[0];
    uint32_t C = in0->shapes[1];
    uint32_t H = in0->shapes[2];
    uint32_t N = B * C * H;

    uint32_t group_x = (N + pipeline->shader_info.local_x - 1) / pipeline->shader_info.local_x;
    uint32_t group_y = 1;
    uint32_t group_z = 1;

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
        vkllm_commands_pipeline(context, commands, tensor, bindings, NULL, constants, group_x, group_y, group_z), err,
        free_bindings_out);

    tensor->access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    tensor->pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

free_bindings_out:
    vkllm_array_ptr_free(bindings);
free_constants_out:
    vkllm_shader_constants_free(constants);
    return err;
}

vkllm_err_t vkllm_op_arg_max_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                      struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);
    return VKLLM_ERR_OK;
}
