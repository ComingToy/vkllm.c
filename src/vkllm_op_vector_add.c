#include "vkllm_op_vector_add.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_pipeline.h"
#include "vkllm_common.h"
#include "vkllm_comp_shaders.h"

vkllm_err_t vkllm_op_vector_add(struct vkllm_context *context, struct vkllm_commands *commands,
                                struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    _CHECK_ARGS(tensor->srcs[0] && tensor->srcs[1]);

    struct vkllm_tensor *in0 = tensor->srcs[0], *in1 = tensor->srcs[1];
    _CHECK_ARGS(in0 && in1);
    _CHECK_ARGS(in0->shapes[0] == 1 && in0->shapes[1] == 1 && in0->shapes[2] == 1);
    _CHECK_ARGS(in1->shapes[0] == 1 && in1->shapes[1] == 1 && in1->shapes[2] == 1);
    _CHECK_ARGS(tensor->shapes[0] == 1 && tensor->shapes[1] == 1 && tensor->shapes[2] == 1);
    _CHECK_ARGS(in0->shapes[2] == in1->shapes[2] && in0->shapes[2] == tensor->shapes[2]);

    const uint8_t *spv_code = __get_vec_add_comp_spv_code();
    const size_t spv_bytes = __get_vec_add_comp_spv_size();

    vkllm_err_t err = VKLLM_ERR_OK;

    struct vkllm_shader_constants *constants = NULL, *specializations = NULL;
    _CHECK(vkllm_shader_constants_new(&constants, 32));
    _CHECK_JUMP(vkllm_shader_constants_new(&specializations, 32), err, create_specialization_fail);

    uint32_t N = tensor->shapes[3];
    vkllm_shader_constants_append(constants, N);

    struct vkllm_pipeline *pipeline = NULL;
    struct vkllm_shader_info shader_info = {
        .binding_count = 3, .push_constant_bytes = constants->bytes, .local_x = 32, .local_y = 1, .local_z = 1};

    _CHECK_JUMP(
        vkllm_pipeline_new(context, commands->device, shader_info, spv_code, spv_bytes, specializations, &pipeline),
        err, create_pipeline_fail);

    struct vkllm_array_ptr *bindings = NULL;
    _CHECK_JUMP(vkllm_array_ptr_new(&bindings, 3), err, create_bindings_fail);
    vkllm_array_ptr_append(bindings, in0);
    vkllm_array_ptr_append(bindings, in1);
    vkllm_array_ptr_append(bindings, tensor);

    uint32_t group_x = (N + 31) / 32;

    _CHECK_JUMP(vkllm_commands_pipeline(context, commands, pipeline, bindings, NULL, constants, group_x, 1, 1), err,
                commands_pipeline_fail);

    return VKLLM_ERR_OK;

commands_pipeline_fail:
    vkllm_array_ptr_free(bindings);
create_bindings_fail:
    vkllm_pipeline_free(context, pipeline);
create_pipeline_fail:
    vkllm_shader_constants_free(specializations);
create_specialization_fail:
    vkllm_shader_constants_free(constants);
    return err;
}
