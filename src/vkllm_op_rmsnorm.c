#include "vkllm_op_rmsnorm.h"
#include "src/vkllm_array.h"
#include "src/vkllm_common.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_pipeline.h"
#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_gpu_device.h"
#include "vkllm_tensor.h"

vkllm_err_t vkllm_op_rmsnorm(struct vkllm_context *context, struct vkllm_commands *commands,
                             struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_RMSNORM);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);

    // in0 is the input tensor with shape (B, C, H, W)
    // in1 is the weight tensor with shape (1, 1, 1, W)
    // Output shape should match in0
    _CHECK_ARGS(in0->shapes[0] == tensor->shapes[0]);
    _CHECK_ARGS(in0->shapes[1] == tensor->shapes[1]);
    _CHECK_ARGS(in0->shapes[2] == tensor->shapes[2]);
    _CHECK_ARGS(in0->shapes[3] == tensor->shapes[3]);
    _CHECK_ARGS(in1->shapes[0] == 1 && in1->shapes[1] == 1 && in1->shapes[2] == 1);
    _CHECK_ARGS(in1->shapes[3] == in0->shapes[3]);

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    struct vkllm_dtype_info in0_dtype_info, in1_dtype_info, dtype_info;
    _CHECK(vkllm_get_dtype_info(in0->dtype, &in0_dtype_info));
    _CHECK(vkllm_get_dtype_info(in1->dtype, &in1_dtype_info));
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));

    uint32_t in0_strides[4], in1_strides[4];
    _DIV4_S(in0->strides, in0_dtype_info.bytes, in0_strides);
    _DIV4_S(in1->strides, in1_dtype_info.bytes, in1_strides);

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 128));

    // Push constants: in_shape, w_shape, power, eps
    vkllm_shader_constants_append_n(constants, in0->shapes, 4);
    vkllm_shader_constants_append_n(constants, in0_strides, 4);
    vkllm_shader_constants_append_n(constants, in1->shapes, 4);
    vkllm_shader_constants_append_n(constants, in1_strides, 4);

    // Default RMS norm parameters: power=2.0, eps=1e-6
    float power = 2.0f;
    float eps = 1e-6f;

    vkllm_shader_constants_append(constants, power);
    vkllm_shader_constants_append(constants, eps);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, free_constants_out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    // RMSNorm processes each row (last dimension) independently
    // Each workgroup processes one row, so we need B*C*H workgroups
    uint32_t num_rows = in0->shapes[0] * in0->shapes[1] * in0->shapes[2];
    uint32_t group_x = 1; // Each workgroup has 512 threads to process one row
    uint32_t group_y = num_rows;
    uint32_t group_z = 1;

    // If num_rows is too large, distribute across y and z dimensions
    uint32_t max_group_y = context->device->vk_physical_dev.properties.limits.maxComputeWorkGroupCount[1];
    if (group_y > max_group_y)
    {
        group_y = 512;
        group_z = (num_rows + group_y - 1) / group_y;
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
