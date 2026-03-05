#include "vkllm_op_copy.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_context.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"
#include "vkllm_tensor.h"

static vkllm_err_t vkllm_op_copy_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                              struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline);
    _CHECK_ARGS(tensor->srcs[0]);

    *pipeline = NULL;

    if (tensor->dtype != tensor->srcs[0]->dtype)
    {
        log_error("copy op requires same dtype for src and dst. src: %s, dst: %s",
                  vkllm_dtype_s(tensor->srcs[0]->dtype), vkllm_dtype_s(tensor->dtype));
        return VKLLM_ERR_ARGS;
    }

    if (tensor->dtype == vkllm_dtype_float16)
    {
        *pipeline = context->pipelines.copy.f16;
    }
    else if (tensor->dtype == vkllm_dtype_float32)
    {
        *pipeline = context->pipelines.copy.f32;
    }
    else
    {
        log_error("unsupported copy dtype: %s", vkllm_dtype_s(tensor->dtype));
        return VKLLM_ERR_ARGS;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_copy_init(struct vkllm_context *context, struct vkllm_commands *commands,
                               struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_COPY);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    _CHECK_ARGS(in0->shapes[0] == tensor->shapes[0] && in0->shapes[1] == tensor->shapes[1] &&
                in0->shapes[2] == tensor->shapes[2] && in0->shapes[3] == tensor->shapes[3]);

    struct vkllm_pipeline *pipeline = NULL;
    _CHECK(vkllm_op_copy_get_pipeline(context, tensor, &pipeline));
    tensor->pipeline = pipeline;
    _CHECK(vkllm_pipeline_alloc_desc_set(context, pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_copy_run(struct vkllm_context *context, struct vkllm_commands *commands,
                              struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0]);

    struct vkllm_tensor *in0 = tensor->srcs[0];

    struct vkllm_dtype_info dtype_info;
    uint32_t strides[4], in0_strides[4];
    _CHECK(vkllm_get_dtype_info(in0->dtype, &dtype_info));
    _DIV4_S(tensor->strides, dtype_info.bytes, strides);
    _DIV4_S(in0->strides, dtype_info.bytes, in0_strides);

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, sizeof(uint32_t) * 32));
    vkllm_shader_constants_append_n(constants, in0->shapes, 4);
    vkllm_shader_constants_append_n(constants, in0_strides, 4);
    vkllm_shader_constants_append_n(constants, tensor->shapes, 4);
    vkllm_shader_constants_append_n(constants, strides, 4);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 2), err, free_constants_out);

    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, tensor);

    struct vkllm_pipeline *pipeline = tensor->pipeline;
    uint32_t N = _MUL4(tensor->shapes);
    uint32_t group_x = (N + pipeline->shader_info.local_x - 1) / pipeline->shader_info.local_x;
    uint32_t group_y = 1, group_z = 1;

    VkPhysicalDeviceLimits *limits = &context->device->vk_physical_dev.properties.limits;
    if (group_x > limits->maxComputeWorkGroupCount[0])
    {
        log_info("group_x %u > %u, ajust to max\n", group_x, limits->maxComputeWorkGroupCount[0]);
        group_x = limits->maxComputeWorkGroupCount[0];
        N = N - group_x * pipeline->shader_info.local_x;
        group_y = (N + pipeline->shader_info.local_y - 1) / pipeline->shader_info.local_y;
        if (group_y > limits->maxComputeWorkGroupCount[1])
        {
            log_info("group_y %u > %u, ajust to max\n", group_y, limits->maxComputeWorkGroupCount[1]);
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

vkllm_err_t vkllm_op_copy_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                   struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);
    return VKLLM_ERR_OK;
}
