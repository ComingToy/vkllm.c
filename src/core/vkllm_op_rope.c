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
    // Fused KV write: params->fused_write marks an output that aliases the
    // KV cache buffer (row offset in params->cache_row).
    bool kv = params->fused_write;
    // Paired Q+K: one dispatch rotates both halves (decode, MHA); the pair
    // pipelines embed the fused KV write for the K side.
    bool pair = params->pair_q && params->q_out != NULL;

    if (in0->dtype == vkllm_dtype_float16)
    {
        // Always fp32 theta math for f16 inputs: fp16 angles lose precision
        // at large positions (see the shader BUILD note). f16f32 is f16
        // storage + fp32 compute.
        if (pair)
            *pipeline = context->pipelines.rope.pair_f16f32[neox_style];
        else
            *pipeline = kv ? context->pipelines.rope.f16f32kv[neox_style]
                           : context->pipelines.rope.f16f32[neox_style];
        return VKLLM_ERR_OK;
    }
    else if (in0->dtype == vkllm_dtype_float32)
    {
        if (pair)
            *pipeline = context->pipelines.rope.pair_f32f32[neox_style];
        else
            *pipeline = kv ? context->pipelines.rope.f32f32kv[neox_style] : context->pipelines.rope.f32f32[neox_style];
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
    _CHECK(vkllm_pipeline_alloc_desc_set(context, tensor->pipeline, &tensor->vk_desc_set));

    return VKLLM_ERR_OK;
}

// Paired Q+K dispatch: srcs = {K_in, Q_in}; the K half of the output aliases
// the KV cache (fused_write semantics) and the Q half writes q_out's own
// buffer. Pushes the 21-uint block the pair shader expects.
static vkllm_err_t vkllm_op_rope_run_pair(struct vkllm_context *context, struct vkllm_commands *commands,
                                          struct vkllm_tensor *tensor)
{
    struct vkllm_op_rope_params *params = (struct vkllm_op_rope_params *)tensor->params;
    struct vkllm_tensor *k_in = tensor->srcs[0];
    struct vkllm_tensor *q_in = tensor->srcs[1];
    struct vkllm_tensor *q_out = params->q_out;

    struct vkllm_dtype_info dtype_info;
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    uint32_t k_s[4], ko_s[4], q_s[4], qo_s[4];
    _DIV4_S(k_in->strides, dtype_info.bytes, k_s);
    _DIV4_S(tensor->strides, dtype_info.bytes, ko_s);
    _DIV4_S(q_in->strides, dtype_info.bytes, q_s);
    _DIV4_S(q_out->strides, dtype_info.bytes, qo_s);

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, k_s, 4);
    vkllm_push_u32_n(&pc, ko_s, 4);
    vkllm_push_u32_n(&pc, q_s, 4);
    vkllm_push_u32_n(&pc, qo_s, 4);
    vkllm_push_u32(&pc, tensor->shapes[1]); // heads (per side)
    vkllm_push_u32(&pc, tensor->shapes[3]); // head_dim
    vkllm_push_u32(&pc, params->offset);
    vkllm_push_f32(&pc, params->base);
    vkllm_push_u32(&pc, params->cache_row);

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 4), err, out);
    vkllm_array_ptr_append(bindings, k_in);
    vkllm_array_ptr_append(bindings, tensor); // K output (aliases the cache)
    vkllm_array_ptr_append(bindings, q_in);
    vkllm_array_ptr_append(bindings, q_out);

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    // K pairs + Q pairs = heads * dim threads.
    uint32_t N = tensor->shapes[1] * tensor->shapes[3];
    uint32_t group_x = 0, group_z = 0;
    _CHECK(vkllm_calc_linear_group_counts(context, N, pipeline->shader_info.local_x, &group_x, &group_z));
    const uint32_t group_y = 1;

    _CHECK_JUMP(
        vkllm_commands_pipeline(context, commands, tensor, bindings, NULL, &pc, group_x, group_y, group_z), err,
        free_bindings_out);

    tensor->access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    tensor->pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    // The Q output is written by this dispatch too — mark it so its
    // consumers see the dependency.
    q_out->access_flags = VK_ACCESS_SHADER_WRITE_BIT;
    q_out->pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
free_bindings_out:
    vkllm_array_ptr_free(bindings);
out:
    return err;
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
    if (params && params->pair_q && params->q_out != NULL)
    {
        return vkllm_op_rope_run_pair(context, commands, tensor);
    }

    uint32_t offset = params ? params->offset : 0;
    float base = params ? params->base : 10000.0f;
    uint32_t cache_row = params ? params->cache_row : 0;

    struct vkllm_dtype_info dtype_info;
    uint32_t out_strides[4], in_strides[4];
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));
    _DIV4_S(in0->strides, dtype_info.bytes, in_strides);
    _DIV4_S(tensor->strides, dtype_info.bytes, out_strides);

    struct vkllm_push_constants pc;
    vkllm_push_constants_reset(&pc);
    vkllm_push_u32_n(&pc, in0->shapes, 4);
    vkllm_push_u32_n(&pc, in_strides, 4);
    vkllm_push_u32_n(&pc, tensor->shapes, 4);
    vkllm_push_u32_n(&pc, out_strides, 4);
    vkllm_push_u32(&pc, offset);
    vkllm_push_f32(&pc, base);
    if (((struct vkllm_op_rope_params *)tensor->params)->fused_write)
    {
        // FUSED_KV pipeline: extra row_offset push constant.
        vkllm_push_u32(&pc, cache_row);
    }

    vkllm_err_t err = VKLLM_ERR_OK;
    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 2), err, out);

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

vkllm_err_t vkllm_op_rope_post_run(struct vkllm_context *context, struct vkllm_commands *commands,
                                   struct vkllm_tensor *tensor)
{
    __UNUSED(context);
    __UNUSED(commands);
    __UNUSED(tensor);
    return VKLLM_ERR_OK;
}
