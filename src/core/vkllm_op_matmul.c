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

// Non-transposed GEMV: M == 1 activations x row-major B [K,N] (K-major
// weights). This is the decode attention-values shape (attn_weights @ V),
// where V rows are K (= cur_len) and columns are the head dim. The tiled
// matmul wastes 63/64 of its A-tile loads here; the dedicated kernel streams
// the B rows coalesced (see vkllm_mat_mul_vec_nt.comp).
static inline bool is_mat_mul_vec_nt(struct vkllm_tensor *tensor)
{
    struct vkllm_tensor *in0 = tensor->srcs[0];
    struct vkllm_tensor *in1 = tensor->srcs[1];

    if (in0->shapes[2] != 1 || matmul_transposed_b(tensor))
        return false;
    // B strictly [B,C,K,N] row-major with unit inner stride.
    struct vkllm_dtype_info in1_dtype_info;
    if (vkllm_get_dtype_info(in1->dtype, &in1_dtype_info) != VKLLM_ERR_OK)
        return false;
    if (in1->strides[3] / in1_dtype_info.bytes != 1)
        return false;
    // Per-head N must be covered by one workgroup sweep (the shader loops
    // n += WorkGroupSize, so any N works; keep the guard on the activation
    // row being single).
    return in0->shapes[2] == 1;
}

// act == 3: dual-weight GEMV. srcs = {x, Wg, Wu}; out last dim = 2*FF.
static bool is_mat_mul_vec_dual(struct vkllm_tensor *tensor)
{
    const struct vkllm_op_matmul_params *params = get_matmul_params(tensor);
    struct vkllm_tensor *in0 = tensor->srcs[0];
    struct vkllm_tensor *wg = tensor->srcs[1];
    struct vkllm_tensor *wu = tensor->srcs[2];

    if (params->act != 3 || in0 == NULL || wg == NULL || wu == NULL)
        return false;
    if (in0->shapes[2] != 1 || !matmul_transposed_b(tensor))
        return false;
    if (wg->shapes[2] != wu->shapes[2] || wg->shapes[3] != wu->shapes[3])
        return false;
    // Strictly contiguous weights (same contract as the single GEMV).
    struct vkllm_dtype_info info;
    if (vkllm_get_dtype_info(wg->dtype, &info) != VKLLM_ERR_OK)
        return false;
    uint32_t K = in0->shapes[3];
    return wg->strides[3] / info.bytes == 1 && wg->strides[2] / info.bytes == K &&
           wu->strides[3] / info.bytes == 1 && wu->strides[2] / info.bytes == K;
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

    // act == 2 selects the fused GEMV + residual-add epilogue (srcs[2] is the
    // residual). Only valid on the GEMV path; the tiled matmul keeps act ∈ {0,1}.
    const struct vkllm_op_matmul_params *params = get_matmul_params(tensor);
    bool fused_add = params->act == 2 && tensor->srcs[2] != NULL;

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
                *pipeline = fused_add ? context->pipelines.mat_mul_vec.f16f16fa[a_boardcast_type][b_boardcast_type]
                                      : context->pipelines.mat_mul_vec.f16f16[a_boardcast_type][b_boardcast_type];
            }
            else if (is_mat_mul_vec_nt(tensor))
            {
                *pipeline = context->pipelines.mat_mul_vec_nt.f16f16;
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
                *pipeline = fused_add ? context->pipelines.mat_mul_vec.f16f32fa[a_boardcast_type][b_boardcast_type]
                                      : context->pipelines.mat_mul_vec.f16f32[a_boardcast_type][b_boardcast_type];
            }
            else if (is_mat_mul_vec_nt(tensor))
            {
                *pipeline = context->pipelines.mat_mul_vec_nt.f16f32;
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
            *pipeline = fused_add ? context->pipelines.mat_mul_vec.f32f32fa[a_boardcast_type][b_boardcast_type]
                                  : context->pipelines.mat_mul_vec.f32f32[a_boardcast_type][b_boardcast_type];
        }
        else if (is_mat_mul_vec_nt(tensor))
        {
            *pipeline = context->pipelines.mat_mul_vec_nt.f32f32;
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
    if (is_mat_mul_vec_dual(tensor))
    {
        return vkllm_op_mat_mul_vec_dual_init(context, commands, tensor);
    }

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
    // Fused residual-add path: srcs[2] is the residual and must share the
    // output layout (the shader indexes it through shape_out).
    struct vkllm_tensor *residual = act == 2 ? tensor->srcs[2] : NULL;
    _CHECK_ARGS(!residual || residual->data.vk_buf != NULL);

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
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, residual ? 4 : 3), err, out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);
    if (residual)
    {
        vkllm_array_ptr_append(bindings, residual);
    }

    // Each subgroup computes a PAIR of adjacent weight rows (activations
    // shared), so one workgroup per row pair: ceil(N/2) along x with
    // overflow into z.
    uint32_t N = _MUL4(tensor->shapes);
    uint32_t group_x = (N + 1) / 2;
    uint32_t group_y = 1, group_z = 0;
    _CHECK(vkllm_calc_linear_group_counts(context, group_x, 1, &group_x, &group_z));

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

vkllm_err_t vkllm_op_mat_mul_vec_nt_run(struct vkllm_context *context, struct vkllm_commands *commands,
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

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, in0->shapes, 4);
    vkllm_push_u32_n(&pc, strides0, 4);
    vkllm_push_u32_n(&pc, in1->shapes, 4);
    vkllm_push_u32_n(&pc, strides1, 4);
    vkllm_push_u32_n(&pc, tensor->shapes, 4);
    vkllm_push_u32_n(&pc, strides2, 4);
    vkllm_push_f32(&pc, params->scale);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    // One workgroup per (b,c) activation row; the shader maps lanes to the
    // N output columns (n = tid, tid + WorkGroupSize, ...).
    uint32_t rows = in0->shapes[0] * in0->shapes[1];
    uint32_t group_x = rows;
    uint32_t group_y = 1, group_z = 1;
    if (group_x > context->device->vk_physical_dev.properties.limits.maxComputeWorkGroupCount[0])
    {
        // Large batches overflow x: split across z (the shader indexes rows
        // linearly via gl_WorkGroupID.x, so this needs the linear helper).
        group_z = (group_x + 511) / 512;
        group_x = 512;
    }

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

vkllm_err_t vkllm_op_mat_mul_vec_dual_init(struct vkllm_context *context, struct vkllm_commands *commands,
                                           struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1] && tensor->srcs[2]);
    _CHECK_ARGS(is_mat_mul_vec_dual(tensor));

    struct vkllm_tensor *in0 = tensor->srcs[0], *wg = tensor->srcs[1], *wu = tensor->srcs[2];
    _CHECK_ARGS(in0->dtype == wg->dtype && wg->dtype == wu->dtype);
    _CHECK_ARGS(in0->shapes[3] == wg->shapes[3] && wg->shapes[3] == wu->shapes[3]);

    if (in0->dtype == vkllm_dtype_float16 && context->device->support_16bit_storage)
        tensor->pipeline = context->pipelines.mat_mul_vec_dual.f16f32;
    else if (in0->dtype == vkllm_dtype_float32)
        tensor->pipeline = context->pipelines.mat_mul_vec_dual.f32f32;
    else
        return VKLLM_ERR_PIPELINE_NOT_FOUND;

    _CHECK(vkllm_pipeline_alloc_desc_set(context, tensor->pipeline, &tensor->vk_desc_set));
    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_op_mat_mul_vec_dual_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);

    struct vkllm_tensor *in0 = tensor->srcs[0], *wg = tensor->srcs[1], *wu = tensor->srcs[2];

    struct vkllm_dtype_info d0, dg, dt;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &d0));
    _CHECK(vkllm_get_dtype_info(wg->dtype, &dg));
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dt));

    uint32_t s0[4], sg[4], st[4];
    _DIV4_S(in0->strides, d0.bytes, s0);
    _DIV4_S(wg->strides, dg.bytes, sg);
    _DIV4_S(tensor->strides, dt.bytes, st);

    const struct vkllm_op_matmul_params *params = get_matmul_params(tensor);

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, in0->shapes, 4);
    vkllm_push_u32_n(&pc, s0, 4);
    vkllm_push_u32_n(&pc, wg->shapes, 4);
    vkllm_push_u32_n(&pc, sg, 4);
    vkllm_push_u32_n(&pc, tensor->shapes, 4);
    vkllm_push_u32_n(&pc, st, 4);
    vkllm_push_f32(&pc, params->scale);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 4), err, out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, wg);
    vkllm_array_ptr_append(bindings, wu);
    vkllm_array_ptr_append(bindings, tensor);

    // One subgroup per output row (2FF rows), overflow into z.
    uint32_t FF = wg->shapes[2];
    uint32_t group_x = 2 * FF;
    uint32_t group_y = 1, group_z = 0;
    _CHECK(vkllm_calc_linear_group_counts(context, group_x, 1, &group_x, &group_z));

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

    if (is_mat_mul_vec_dual(tensor))
    {
        return vkllm_op_mat_mul_vec_dual_run(context, commands, tensor);
    }

    if (is_mat_mul_vec(tensor))
    {
        return vkllm_op_mat_mul_vec_run(context, commands, tensor);
    }

    if (is_mat_mul_vec_nt(tensor))
    {
        return vkllm_op_mat_mul_vec_nt_run(context, commands, tensor);
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
