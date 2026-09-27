#include "vkllm_op_matmul.h"
#include "vkllm_array.h"
#include "vkllm_commands.h"
#include "vkllm_common.h"
#include "vkllm_dtypes.h"
#include "vkllm_gpu_device.h"
#include "vkllm_pipeline.h"

static const struct vkllm_op_matmul_params *get_matmul_params(const struct vkllm_tensor *tensor)
{
    return (const struct vkllm_op_matmul_params *)tensor->params;
}

// Prefer the explicitly declared layout. Fall back to shape inference only
// when the caller left transposed_b unspecified (== -1), so existing callers
// using positional initializers keep their old behavior.
static int matmul_transposed_b(const struct vkllm_tensor *tensor)
{
    const struct vkllm_op_matmul_params *params = get_matmul_params(tensor);
    if (params->transposed_b == 0 || params->transposed_b == 1)
    {
        return (int)params->transposed_b;
    }

    struct vkllm_tensor *a = tensor->srcs[0];
    struct vkllm_tensor *b = tensor->srcs[1];

    uint32_t M = tensor->shapes[2];
    uint32_t N = tensor->shapes[3];
    uint32_t K = a->shapes[3];

    return a->shapes[2] == M && b->shapes[2] == N && a->shapes[3] == K && b->shapes[3] == K;
}

// GEMV requires a single activation row AND strictly contiguous fp16
// weights: row pitch exactly K elements and K % 4 == 0 (the shader's
// vectorized pair loads need both rows 4-element aligned; the fallback
// scalar path inside the shader mis-indexes the second row on some
// drivers when rows are unaligned — route those to the tiled matmul
// instead, which is verified for all shapes).
static bool gemv_weights_ok(const struct vkllm_tensor *tensor)
{
    const struct vkllm_tensor *in0 = tensor->srcs[0];
    const struct vkllm_tensor *in1 = tensor->srcs[1];

    struct vkllm_dtype_info in1_dtype_info;
    if (vkllm_get_dtype_info(in1->dtype, &in1_dtype_info) != VKLLM_ERR_OK)
        return false;

    uint32_t K = in0->shapes[3];
    uint32_t row_pitch = in1->strides[2] / in1_dtype_info.bytes;

    return in1->strides[3] / in1_dtype_info.bytes == 1 && row_pitch == K && (K & 3u) == 0;
}

static inline bool is_mat_mul_vec(struct vkllm_tensor *tensor)
{
    struct vkllm_tensor *in0 = tensor->srcs[0];
    return in0->shapes[2] == 1 && matmul_transposed_b(tensor) && gemv_weights_ok(tensor);
}

#define BOARDCAST_AXIS0 0
#define BOARDCAST_AXIS1 1
#define BOARDCAST_AXIS01 2
#define BOARDCAST_NONE 3

static int boardcast_type(struct vkllm_tensor *a, struct vkllm_tensor *b)
{
    int type = BOARDCAST_NONE;

    if (a->shapes[0] == 1 && a->shapes[1] != 1 && b->shapes[0] != 1)
    {
        type = BOARDCAST_AXIS0;
    }
    else if (a->shapes[0] != 1 && a->shapes[1] == 1 && b->shapes[1] != 1)
    {
        type = BOARDCAST_AXIS1;
    }
    else if (a->shapes[0] == 1 && a->shapes[1] == 1 && b->shapes[0] != 1 && b->shapes[1] != 1)
    {
        type = BOARDCAST_AXIS01;
    }
    else if (a->shapes[0] == 1 && a->shapes[1] == 1 && b->shapes[0] != 1 && b->shapes[1] == 1)
    {
        type = BOARDCAST_AXIS0;
    }
    else if (a->shapes[0] == 1 && a->shapes[1] == 1 && b->shapes[0] == 1 && b->shapes[1] != 1)
    {
        type = BOARDCAST_AXIS1;
    }

    return type;
}

static vkllm_err_t vkllm_op_matmul_get_pipeline(struct vkllm_context *context, struct vkllm_tensor *tensor,
                                                struct vkllm_pipeline **pipeline)
{
    _CHECK_ARGS(context && tensor && pipeline);
    *pipeline = NULL;

    int tranposed_b = matmul_transposed_b(tensor);
    int a_boardcast_type = boardcast_type(tensor->srcs[0], tensor->srcs[1]);
    int b_boardcast_type = boardcast_type(tensor->srcs[1], tensor->srcs[0]);

    struct vkllm_tensor *in0 = tensor->srcs[0];
    struct vkllm_tensor *in1 = tensor->srcs[1];

    if (in0->dtype == vkllm_dtype_float16 && in1->dtype == vkllm_dtype_float16)
    {
        if (tensor->dtype != vkllm_dtype_float16)
        {
            log_error("input dtype = fp16 and output dtype != fp16");
            return VKLLM_ERR_PIPELINE_NOT_FOUND;
        }

        if (!context->device->support_16bit_storage)
        {
            log_error("matmul pipeline: fp16 type inputs is unsupported.");
            return VKLLM_ERR_PIPELINE_NOT_FOUND;
        }

        if (context->device->support_fp16_arithmetic)
        {
            if (is_mat_mul_vec(tensor))
            {
                *pipeline = context->pipelines.mat_mul_vec.f16f16[a_boardcast_type][b_boardcast_type];
            }
            else
            {
                *pipeline = context->pipelines.matmul.f16f16f16[a_boardcast_type][b_boardcast_type][tranposed_b];
            }
        }
        else
        {
            if (is_mat_mul_vec(tensor))
            {
                *pipeline = context->pipelines.mat_mul_vec.f16f32[a_boardcast_type][b_boardcast_type];
            }
            else
            {
                *pipeline = context->pipelines.matmul.f16f32f16[a_boardcast_type][b_boardcast_type][tranposed_b];
            }
        }
    }
    else if (in0->dtype == vkllm_dtype_float32 && in1->dtype == vkllm_dtype_float32)
    {
        if (is_mat_mul_vec(tensor))
        {
            *pipeline = context->pipelines.mat_mul_vec.f32f32[a_boardcast_type][b_boardcast_type];
        }
        else
        {
            *pipeline = context->pipelines.matmul.f32f32f32[a_boardcast_type][b_boardcast_type][tranposed_b];
        }
    }
    else
    {
        return VKLLM_ERR_PIPELINE_NOT_FOUND;
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_matmul_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                 struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_MATMUL);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);
    _CHECK(vkllm_op_matmul_get_pipeline(context, tensor, &tensor->pipeline));
    _CHECK(vkllm_pipeline_alloc_desc_set(context, tensor->pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

static vkllm_err_t vkllm_op_mat_mul_vec_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                            struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_MATMUL);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);

    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    struct vkllm_dtype_info in0_dtype_info;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &in0_dtype_info));

    struct vkllm_dtype_info in1_dtype_info;
    _CHECK(vkllm_get_dtype_info(in1->dtype, &in1_dtype_info));

    uint32_t strides0[4], strides1[4], strides2[4];
    _DIV4_S(in0->strides, in0_dtype_info.bytes, strides0);
    _DIV4_S(in1->strides, in1_dtype_info.bytes, strides1);
    _DIV4_S(tensor->strides, dtype_info.bytes, strides2);

    const struct vkllm_op_matmul_params *params = get_matmul_params(tensor);
    const float scale = params->scale;
    const int32_t act = params->act;

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, in0->shapes, 4);
    vkllm_push_u32_n(&pc, strides0, 4);
    vkllm_push_u32_n(&pc, in1->shapes, 4);
    vkllm_push_u32_n(&pc, strides1, 4);
    vkllm_push_u32_n(&pc, tensor->shapes, 4);
    vkllm_push_u32_n(&pc, strides2, 4);
    vkllm_push_f32(&pc, scale);
    vkllm_push_u32(&pc, act);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    // Each subgroup computes a PAIR of adjacent weight rows (activations
    // shared), so one workgroup per row pair: ceil(N/2) along x with
    // overflow into z.
    uint32_t N = _MUL4(tensor->shapes);
    uint32_t group_x = (N + 1) / 2;
    uint32_t group_y = 1, group_z = 0;
    _CHECK(vkllm_calc_linear_group_counts(context, group_x, 1, &group_x, &group_z));
    const uint32_t local_y = 1;

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

vkllm_err_t vkllm_op_mat_mul_mat_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                     struct vkllm_tensor *tensor)
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

    uint32_t B = tensor->shapes[0];
    uint32_t C = tensor->shapes[1];
    uint32_t M = tensor->shapes[2];
    uint32_t N = tensor->shapes[3];
    uint32_t BATCH = B * C;

    _CHECK_ARGS(in0->shapes[2] == M);

    uint32_t K = in0->shapes[3];

    if (!matmul_transposed_b(tensor))
    {
        if (!(in1->shapes[3] == N && in1->shapes[2] == K))
        {
            log_error("matmul %s shape mismatch: in0 [%u,%u,%u,%u], in1 [%u,%u,%u,%u], out [%u,%u,%u,%u], "
                      "expected in1 == [%u,%u,%u,%u]",
                      tensor->name, in0->shapes[0], in0->shapes[1], in0->shapes[2], in0->shapes[3], in1->shapes[0],
                      in1->shapes[1], in1->shapes[2], in1->shapes[3], tensor->shapes[0], tensor->shapes[1],
                      tensor->shapes[2], tensor->shapes[3], in1->shapes[0], in1->shapes[1], K, N);
            return VKLLM_ERR_ARGS;
        }
    }

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    struct vkllm_dtype_info in0_dtype_info;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &in0_dtype_info));

    struct vkllm_dtype_info in1_dtype_info;
    _CHECK(vkllm_get_dtype_info(in1->dtype, &in1_dtype_info));

    uint32_t in0_stride = in0->strides[2] / in0_dtype_info.bytes;
    uint32_t in1_stride = in1->strides[2] / in1_dtype_info.bytes;
    uint32_t out0_stride = tensor->strides[2] / dtype_info.bytes;

    uint32_t in0_bstride = in0->strides[0] / in0_dtype_info.bytes;
    uint32_t in1_bstride = in1->strides[0] / in1_dtype_info.bytes;
    uint32_t out0_bstride = tensor->strides[0] / dtype_info.bytes;

    uint32_t in0_cstride = in0->strides[1] / in0_dtype_info.bytes;
    uint32_t in1_cstride = in1->strides[1] / in1_dtype_info.bytes;
    uint32_t out0_cstride = tensor->strides[1] / dtype_info.bytes;

    const struct vkllm_op_matmul_params *params = get_matmul_params(tensor);
    const float scale = params->scale;
    const int32_t act = params->act;

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32(&pc, in0_stride);
    vkllm_push_u32(&pc, in1_stride);
    vkllm_push_u32(&pc, out0_stride);
    vkllm_push_u32(&pc, in0_bstride);
    vkllm_push_u32(&pc, in1_bstride);
    vkllm_push_u32(&pc, out0_bstride);
    vkllm_push_u32(&pc, in0_cstride);
    vkllm_push_u32(&pc, in1_cstride);
    vkllm_push_u32(&pc, out0_cstride);
    vkllm_push_u32(&pc, B);
    vkllm_push_u32(&pc, C);
    vkllm_push_u32(&pc, M);
    vkllm_push_u32(&pc, N);
    vkllm_push_u32(&pc, K);
    vkllm_push_f32(&pc, scale);
    vkllm_push_u32(&pc, act);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    // Based on the shader code from vkllm_matmul.comp:
    // BM = 128, BN = 128, workgroup size = (1, 1, 256)
    // Each workgroup processes a BM x BN block of the output
    uint32_t BM = 64;
    uint32_t BN = 64;
    uint32_t group_x = (M + BM - 1) / BM;
    uint32_t group_y = (N + BN - 1) / BN;
    uint32_t group_z = BATCH;

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

vkllm_err_t vkllm_op_matmul_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                struct vkllm_tensor *tensor)
{
    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);

    if (is_mat_mul_vec(tensor))
    {
        return vkllm_op_mat_mul_vec_run(context, commands, tensor);
    }

    return vkllm_op_mat_mul_mat_run(context, commands, tensor);
}

vkllm_err_t vkllm_op_matmul_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                     struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);

    return VKLLM_ERR_OK;
}
