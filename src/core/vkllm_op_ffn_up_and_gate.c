#include "vkllm_op_ffn_up_and_gate.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

static vkllm_err_t vkllm_op_ffn_up_and_gate_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                                         struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline && tensor->srcs[0] && tensor->srcs[1] && tensor->srcs[2]);
    *pipeline = NULL;

    if (tensor->srcs[0]->dtype == vkllm_dtype_float16 && tensor->srcs[1]->dtype == vkllm_dtype_float16 &&
        tensor->srcs[2]->dtype == vkllm_dtype_float16)
    {
        if (!context->device->support_16bit_storage)
        {
            log_error("ffn_up_and_gate pipeline: fp16 type inputs is unsupported.");
            return VKLLM_ERR_PIPELINE_NOT_FOUND;
        }

        if (tensor->dtype == vkllm_dtype_float16)
        {
            if (context->device->support_fp16_arithmetic)
            {
                *pipeline = context->pipelines.ffn.f16f32f16;
            }
            else
            {
                *pipeline = context->pipelines.ffn.f16f32f16;
            }
        }
        else if (tensor->dtype == vkllm_dtype_float32)
        {
            *pipeline = context->pipelines.ffn.f16f32f32;
        }
        else
        {
            log_error("ffn_up_and_gate: input dtype = fp16, unsupported output dtype: %s",
                      vkllm_dtype_s(tensor->dtype));
            return VKLLM_ERR_PIPELINE_NOT_FOUND;
        }
    }
    else if (tensor->srcs[0]->dtype == vkllm_dtype_float32 && tensor->srcs[1]->dtype == vkllm_dtype_float32 &&
             tensor->srcs[2]->dtype == vkllm_dtype_float32)
    {
        if (tensor->dtype != vkllm_dtype_float32)
        {
            log_error("ffn_up_and_gate: input dtype = fp32, unsupported output dtype: %s",
                      vkllm_dtype_s(tensor->dtype));
            return VKLLM_ERR_PIPELINE_NOT_FOUND;
        }
        *pipeline = context->pipelines.ffn.f32f32f32;
    }
    else
    {
        log_error("ffn_up_and_gate unsupported dtype combination: in0=%s, in1=%s, in2=%s, out=%s",
                  vkllm_dtype_s(tensor->srcs[0]->dtype), vkllm_dtype_s(tensor->srcs[1]->dtype),
                  vkllm_dtype_s(tensor->srcs[2]->dtype), vkllm_dtype_s(tensor->dtype));
        return VKLLM_ERR_PIPELINE_NOT_FOUND;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_ffn_up_and_gate_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1] && tensor->srcs[2]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_FFN_UP_AND_GATE);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    struct vkllm_tensor *in1 = tensor->srcs[1];
    struct vkllm_tensor *in2 = tensor->srcs[2];
    _CHECK_ARGS(in0 && in1 && in2);

    struct vkllm_pipeline *pipeline = NULL;
    _CHECK(vkllm_op_ffn_up_and_gate_get_pipeline(context, tensor, &pipeline));
    tensor->pipeline = pipeline;
    _CHECK(vkllm_pipeline_alloc_desc_set(context, tensor->pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_ffn_up_and_gate_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                         struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1] && tensor->srcs[2]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_FFN_UP_AND_GATE);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    struct vkllm_tensor *in1 = tensor->srcs[1];
    struct vkllm_tensor *in2 = tensor->srcs[2];
    _CHECK_ARGS(in0 && in1 && in2);

    uint32_t B = tensor->shapes[0];
    uint32_t C = tensor->shapes[1];
    uint32_t M = tensor->shapes[2];
    uint32_t N = tensor->shapes[3];

    _CHECK_ARGS(in0->shapes[2] == M);
    _CHECK_ARGS(in1->shapes[2] == N);
    _CHECK_ARGS(in2->shapes[2] == N);

    uint32_t K = in0->shapes[3];
    _CHECK_ARGS(in1->shapes[3] == K);
    _CHECK_ARGS(in2->shapes[3] == K);

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    struct vkllm_dtype_info in0_dtype_info;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &in0_dtype_info));

    struct vkllm_dtype_info in1_dtype_info;
    _CHECK(vkllm_get_dtype_info(in1->dtype, &in1_dtype_info));

    struct vkllm_dtype_info in2_dtype_info;
    _CHECK(vkllm_get_dtype_info(in2->dtype, &in2_dtype_info));

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 96));
    vkllm_shader_constants_append_n(constants, in0->shapes, 4);
    vkllm_shader_constants_append_n(constants, in1->shapes, 4);
    vkllm_shader_constants_append_n(constants, in2->shapes, 4);
    vkllm_shader_constants_append_n(constants, tensor->shapes, 4);
    vkllm_shader_constants_append_n(constants, in0->strides, 4);
    vkllm_shader_constants_append_n(constants, in1->strides, 4);
    vkllm_shader_constants_append_n(constants, in2->strides, 4);
    vkllm_shader_constants_append_n(constants, tensor->strides, 4);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 4), err, free_constants_out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, in2);
    vkllm_array_ptr_append(bindings, tensor);

    uint32_t BM = 128;
    uint32_t BN = 128;
    uint32_t group_x = (M + BM - 1) / BM;
    uint32_t group_y = (N + BN - 1) / BN;
    uint32_t group_z = B * C;

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

vkllm_err_t vkllm_op_ffn_up_and_gate_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                              struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);

    return VKLLM_ERR_OK;
}
