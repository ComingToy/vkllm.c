#include "vkllm_op_add.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_pipeline.h"
#include "vkllm_common.h"

vkllm_err_t vkllm_op_add(struct vkllm_context *context, struct vkllm_commands *commands, struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_ADD);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);
    _CHECK_ARGS(in0->shapes[0] == in1->shapes[0] && in0->shapes[0] == tensor->shapes[0]);
    _CHECK_ARGS(in0->shapes[1] == in1->shapes[1] && in0->shapes[1] == tensor->shapes[1]);
    _CHECK_ARGS(in0->shapes[2] == in1->shapes[2] && in0->shapes[2] == tensor->shapes[2]);
    _CHECK_ARGS(in0->shapes[3] == in1->shapes[3] && in0->shapes[3] == tensor->shapes[3]);

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    struct vkllm_dtype_info dtype_info;
    uint32_t strides[4];
    _CHECK(vkllm_get_dtype_info(tensor->dtype, &dtype_info));
    _DIV4_S(tensor->strides, dtype_info.bytes, strides);

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 32));
    vkllm_shader_constants_append_n(constants, tensor->shapes, 4);
    vkllm_shader_constants_append_n(constants, strides, 4);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, free_constants_out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    uint32_t N = _MUL4(tensor->shapes);
    uint32_t group_x = (N + pipeline->shader_info.local_x - 1) / pipeline->shader_info.local_x;
    uint32_t group_y = 1;
    uint32_t group_z = 1;

    _CHECK_JUMP(
        vkllm_commands_pipeline(context, commands, pipeline, bindings, NULL, constants, group_x, group_y, group_z), err,
        free_bindings_out);

free_bindings_out:
    vkllm_array_ptr_free(bindings);
free_constants_out:
    vkllm_shader_constants_free(constants);
    return err;
}
