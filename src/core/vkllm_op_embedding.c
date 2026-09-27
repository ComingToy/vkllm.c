#include "vkllm_op_embedding.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_context.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"
#include "vkllm_tensor.h"

static vkllm_err_t vkllm_op_embedding_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                                   struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);

    *pipeline = NULL;
    if ((tensor->dtype != vkllm_dtype_float32 && tensor->dtype != vkllm_dtype_float16) ||
        tensor->srcs[0]->dtype != vkllm_dtype_uint32)
    {
        log_error("unsupported op result dtype: %s, dtype of in0: %s", vkllm_dtype_s(tensor->dtype),
                  vkllm_dtype_s(tensor->srcs[0]->dtype));
        return VKLLM_ERR_ARGS;
    }

    if (tensor->srcs[1]->dtype == vkllm_dtype_float16)
    {
        *pipeline = context->pipelines.embedding.f16;
    }
    else if (tensor->srcs[1]->dtype == vkllm_dtype_float32)
    {
        *pipeline = context->pipelines.embedding.f32;
    }
    else
    {
        *pipeline = NULL;
        return VKLLM_ERR_PIPELINE_NOT_FOUND;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_embedding_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                    struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];

    _CHECK_ARGS(in0->dtype == vkllm_dtype_uint32);
    _CHECK_ARGS(in1->dtype == vkllm_dtype_float16 || in1->dtype == vkllm_dtype_float32);
    _CHECK_ARGS(in0->shapes[0] == 1 && in1->shapes[0] == 1 && in1->shapes[1] == 1);

    _CHECK(vkllm_op_embedding_get_pipeline(context, tensor, &tensor->pipeline));
    _CHECK(vkllm_pipeline_alloc_desc_set(context, tensor->pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_embedding_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                   struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];

    _CHECK_ARGS(in0->dtype == vkllm_dtype_uint32);
    _CHECK_ARGS(in1->dtype == vkllm_dtype_float16 || in1->dtype == vkllm_dtype_float32);
    _CHECK_ARGS(in0->shapes[0] == 1 && in1->shapes[0] == 1 && in1->shapes[1] == 1);

    struct vkllm_dtype_info in0_dtype_info, in1_dtype_info, dtype_info;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &in0_dtype_info));
    _CHECK(vkllm_get_dtype_info(in1->dtype, &in1_dtype_info));
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    uint32_t in0_strides[4], in1_strides[4], strides[4];
    _DIV4_S(in0->strides, in0_dtype_info.bytes, in0_strides);
    _DIV4_S(in1->strides, in1_dtype_info.bytes, in1_strides);
    _DIV4_S(tensor->strides, dtype_info.bytes, strides);

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, in0->shapes, 4);
    vkllm_push_u32_n(&pc, in0_strides, 4);
    vkllm_push_u32_n(&pc, in1->shapes, 4);
    vkllm_push_u32_n(&pc, in1_strides, 4);
    vkllm_push_u32_n(&pc, tensor->shapes, 4);
    vkllm_push_u32_n(&pc, strides, 4);

    uint32_t unk_tok = *(uint32_t *)tensor->params;
    vkllm_push_u32(&pc, unk_tok);

    struct vkllm_array_ptr *bindings = NULL;
    vkllm_array_ptr_new(&bindings, 3);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    struct vkllm_pipeline *pipeline = tensor->pipeline;
    uint32_t N = _MUL4(in0->shapes);
    uint32_t group_x = 0, group_z = 0;
    vkllm_calc_linear_group_counts(context, N, pipeline->shader_info.local_x, &group_x, &group_z);
    vkllm_err_t err = vkllm_commands_pipeline(context, commands, tensor, bindings, NULL, &pc, group_x, 1, group_z);
    vkllm_array_ptr_free(bindings);
    tensor->access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    tensor->pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    return err;
}

vkllm_err_t vkllm_op_embedding_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                        struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);

    return VKLLM_ERR_OK;
}
