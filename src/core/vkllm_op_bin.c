#include "vkllm_op_bin.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

static vkllm_err_t vkllm_op_bin_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                             struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline && tensor->srcs[0] && tensor->srcs[1]);
    *pipeline = NULL;

    const int32_t op = *(const int32_t *)tensor->params;
    if (tensor->dtype != vkllm_dtype_float32 && tensor->dtype != vkllm_dtype_float16)
    {
        log_error("tensor: %s unsupported op result dtype: %s", tensor->name, vkllm_dtype_s(tensor->dtype));
        return VKLLM_ERR_ARGS;
    }

    if (tensor->srcs[0]->dtype == vkllm_dtype_float16 && tensor->srcs[1]->dtype == vkllm_dtype_float16)
    {
        if (context->device->support_fp16_arithmetic)
        {
            *pipeline = context->pipelines.bin.f16f16f16[op];
        }
        else
        {
            *pipeline = context->pipelines.bin.f16f32f16[op];
        }
        return VKLLM_ERR_OK;
    }
    else if (tensor->srcs[0]->dtype == vkllm_dtype_float32 && tensor->srcs[1]->dtype == vkllm_dtype_float32)
    {
        *pipeline = context->pipelines.bin.f32f32f32[op];
        return VKLLM_ERR_OK;
    }
    else
    {
        return VKLLM_ERR_ARGS;
    }
}

vkllm_err_t vkllm_op_bin_init(struct vkllm_context *context, struct vkllm_commands *commands,
                              struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_BIN);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);
    _CHECK_ARGS(in0->shapes[0] == in1->shapes[0] && in0->shapes[0] == tensor->shapes[0]);
    _CHECK_ARGS(in0->shapes[1] == in1->shapes[1] && in0->shapes[1] == tensor->shapes[1]);
    _CHECK_ARGS(in0->shapes[2] == in1->shapes[2] && in0->shapes[2] == tensor->shapes[2]);
    _CHECK_ARGS(in0->shapes[3] == in1->shapes[3] && in0->shapes[3] == tensor->shapes[3]);

    struct vkllm_pipeline *pipeline = NULL;
    _CHECK(vkllm_op_bin_get_pipeline(context, tensor, &pipeline));
    tensor->pipeline = pipeline;
    _CHECK(vkllm_pipeline_alloc_desc_set(context, pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_bin_run(struct vkllm_context *context, struct vkllm_commands *commands,
                             struct vkllm_tensor *tensor)
{

    struct vkllm_dtype_info dtype_info;
    uint32_t strides[4];
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));
    _DIV4_S(tensor->strides, dtype_info.bytes, strides);

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, tensor->shapes, 4);
    vkllm_push_u32_n(&pc, strides, 4);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, out);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    struct vkllm_pipeline *pipeline = tensor->pipeline;
    uint32_t N = _MUL4(tensor->shapes);
    uint32_t group_x = 0, group_z = 0;
    _CHECK(vkllm_calc_linear_group_counts(context, N, pipeline->shader_info.local_x, &group_x, &group_z));
    const uint32_t group_y = 1;

    _CHECK_JUMP(
        vkllm_commands_pipeline(context, commands, tensor, bindings, NULL, &pc, group_x, group_y, group_z), err,
        free_bindings_out);

    tensor->access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    tensor->pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

free_bindings_out:
    vkllm_array_ptr_free(bindings);
out:
    return err;
}

extern vkllm_err_t vkllm_op_bin_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);
    return VKLLM_ERR_OK;
}
