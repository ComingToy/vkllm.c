#include "vkllm_op_vector_add.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_pipeline.h"
#include "vkllm_common.h"

vkllm_err_t vkllm_op_vector_add(struct vkllm_context *context, struct vkllm_commands *commands,
                                struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);
    _CHECK_ARGS(tensor->op == VKLLM_OP_ADD);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);
    _CHECK_ARGS(in0->shapes[0] == 1 && in0->shapes[1] == 1 && in0->shapes[2] == 1);
    _CHECK_ARGS(in1->shapes[0] == 1 && in1->shapes[1] == 1 && in1->shapes[2] == 1);
    _CHECK_ARGS(tensor->shapes[0] == 1 && tensor->shapes[1] == 1 && tensor->shapes[2] == 1);
    _CHECK_ARGS(in0->shapes[2] == in1->shapes[2] && in0->shapes[2] == tensor->shapes[2]);

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_pipeline *pipeline = tensor->pipeline;

    struct vkllm_shader_constants *constants = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 32));

    uint32_t N = tensor->shapes[3];
    vkllm_shader_constants_append(constants, N);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, free_constants_out);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    uint32_t group_x = (N + pipeline->shader_info.local_x - 1) / pipeline->shader_info.local_x;

    _CHECK_JUMP(vkllm_commands_pipeline(context, commands, pipeline, bindings, NULL, constants, group_x, 1, 1), err,
                free_bindings_out);

free_bindings_out:
    vkllm_array_ptr_free(bindings);
free_constants_out:
    vkllm_shader_constants_free(constants);
    return err;
}
