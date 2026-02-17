#include "vkllm_op_embedding.h"
#include "src/vkllm_array.h"
#include "src/vkllm_common.h"
#include "src/vkllm_pipeline.h"
#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

vkllm_err_t vkllm_op_embedding(struct vkllm_context *context, struct vkllm_commands *commands,
                               struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];

    _CHECK_ARGS(in0->dtype == vkllm_dtype_uint32);
    _CHECK_ARGS(in1->dtype == vkllm_dtype_float16);
    _CHECK_ARGS(in0->shapes[0] == 1 && in1->shapes[0] == 1 && in1->shapes[1] == 1);

    struct vkllm_shader_constants *constants = NULL;
    vkllm_shader_constants_new(&constants, 128);
    vkllm_shader_constants_append_n(constants, in0->shapes, 4);
    vkllm_shader_constants_append_n(constants, in0->strides, 4);
    vkllm_shader_constants_append_n(constants, in1->shapes, 4);
    vkllm_shader_constants_append_n(constants, in1->strides, 4);
    vkllm_shader_constants_append_n(constants, tensor->shapes, 4);
    vkllm_shader_constants_append_n(constants, tensor->strides, 4);

    uint32_t unk_tok = *(uint32_t *)tensor->params;
    vkllm_shader_constants_append(constants, unk_tok);

    struct vkllm_array_ptr *bindings = NULL;
    vkllm_array_ptr_new(&bindings, 3);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    struct vkllm_pipeline *pipeline = tensor->pipeline;
    uint32_t N = tensor->shapes[0] * tensor->shapes[1] * tensor->shapes[2] * tensor->shapes[3];
    uint32_t group_x = (N + pipeline->shader_info.local_x - 1) / pipeline->shader_info.local_x;
    vkllm_err_t err = vkllm_commands_pipeline(context, commands, pipeline, bindings, NULL, constants, group_x, 1, 1);

    vkllm_shader_constants_free(constants);
    vkllm_array_ptr_free(bindings);
    return err;
}
