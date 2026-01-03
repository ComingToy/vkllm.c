#include "vkllm_op_matmul.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

static bool is_transposed_b(struct vkllm_tensor *tensor)
{
    struct vkllm_tensor *a = tensor->srcs[0];
    struct vkllm_tensor *b = tensor->srcs[1];

    uint32_t M = tensor->shapes[2];
    uint32_t N = tensor->shapes[3];
    uint32_t K = a->shapes[3];

    return a->shapes[2] == M && b->shapes[2] == N && a->shapes[3] == K && b->shapes[3] == K;
}

static vkllm_err_t vkllm_op_matmul_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                                struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline);
    *pipeline = NULL;

    if (tensor->dtype != vkllm_dtype_float32)
    {
        log_error("tensor %s op = %s, dtype = %s, pipeline not found.", tensor->name, vkllm_op_s(tensor->op),
                  vkllm_dtype_s(tensor->dtype));
        return VKLLM_ERR_PIPELINE_NOT_FOUND;
    }

    bool tranposed_b = is_transposed_b(tensor);
    if (tensor->srcs[0]->dtype == vkllm_dtype_float16 && tensor->srcs[0]->dtype == vkllm_dtype_float16)
    {
        if (!context->device->support_16bit_storage)
        {
            log_error("matmul pipeline: fp16 type inputs is unsupported.");
            return VKLLM_ERR_PIPELINE_NOT_FOUND;
        }

        if (context->device->support_fp16_arithmetic)
        {
            *pipeline = tranposed_b ? context->pipelines.matmul.b0t1f16f16f32 : context->pipelines.matmul.b0t0f16f16f32;
        }
        else
        {
            *pipeline = tranposed_b ? context->pipelines.matmul.b0t1f16f32f32 : context->pipelines.matmul.b0t0f16f32f32;
        }
    }
    else if (tensor->srcs[0]->dtype == vkllm_dtype_float32 && tensor->srcs[0]->dtype == vkllm_dtype_float32)
    {
        *pipeline = tranposed_b ? context->pipelines.matmul.b0t1f32f32f32 : context->pipelines.matmul.b0t0f32f32f32;
    }
    else
    {
        return VKLLM_ERR_PIPELINE_NOT_FOUND;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_matmul(struct vkllm_context *context, struct vkllm_commands *commands, struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_MATMUL);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);

    // For matmul: A[M, K] * B[K, N] = C[M, N]
    // Using 4D tensor format: [1, 1, M, K/N]
    // in0 shape: [1, 1, M, K]
    // in1 shape: [1, 1, K, N]
    // output shape: [1, 1, M, N]

    uint32_t M = tensor->shapes[2];
    uint32_t N = tensor->shapes[3];
    uint32_t BATCH = tensor->shapes[0] * tensor->shapes[1];

    _CHECK_ARGS(in0->shapes[2] == M);

    uint32_t K = in0->shapes[3];

    if (!is_transposed_b(tensor))
    {
        _CHECK_ARGS(in1->shapes[3] == N && in1->shapes[2] == K);
    }

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_pipeline *pipeline = NULL;

    _CHECK(vkllm_op_matmul_get_pipeline(context, tensor, &pipeline));
    tensor->pipeline = pipeline;

    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    struct vkllm_dtype_info in0_dtype_info;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &in0_dtype_info));

    struct vkllm_dtype_info in1_dtype_info;
    _CHECK(vkllm_get_dtype_info(in1->dtype, &in1_dtype_info));

    uint32_t in0_stride = in0->strides[2] / in0_dtype_info.bytes;
    uint32_t in1_stride = in1->strides[2] / in1_dtype_info.bytes;
    uint32_t out0_stride = tensor->strides[2] / dtype_info.bytes;

    uint32_t in0_bstride = in0->strides[1] / in0_dtype_info.bytes;
    uint32_t in1_bstride = in1->strides[1] / in1_dtype_info.bytes;
    uint32_t out0_bstride = tensor->strides[1] / dtype_info.bytes;

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 24));
    vkllm_shader_constants_append(constants, in0_stride);
    vkllm_shader_constants_append(constants, in1_stride);
    vkllm_shader_constants_append(constants, out0_stride);
    vkllm_shader_constants_append(constants, in0_bstride);
    vkllm_shader_constants_append(constants, in1_bstride);
    vkllm_shader_constants_append(constants, out0_bstride);
    vkllm_shader_constants_append(constants, M);
    vkllm_shader_constants_append(constants, N);
    vkllm_shader_constants_append(constants, K);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, free_constants_out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    // Based on the shader code from vkllm_matmul.comp:
    // BM = 128, BN = 128, workgroup size = (1, 1, 256)
    // Each workgroup processes a BM x BN block of the output
    uint32_t BM = 128;
    uint32_t BN = 128;
    uint32_t group_x = (M + BM - 1) / BM;
    uint32_t group_y = (N + BN - 1) / BN;
    uint32_t group_z = BATCH;

    _CHECK_JUMP(
        vkllm_commands_pipeline(context, commands, pipeline, bindings, NULL, constants, group_x, group_y, group_z), err,
        free_bindings_out);

free_bindings_out:
    vkllm_array_ptr_free(bindings);
free_constants_out:
    vkllm_shader_constants_free(constants);
    return err;
}
