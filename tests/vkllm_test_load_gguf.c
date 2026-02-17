#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_errors.h"
#include "src/core/vkllm_tensor.h"
#include "src/models/vkllm_models_llama2.h"

int main(const int argc, const char *argv[])
{
    if (argc != 2)
    {
        log_error("usage: %s <path to gguf>", argv[0]);
        return -1;
    }

    struct vkllm_context *context = NULL;
    struct vkllm_models_llama2 model;

    vkllm_err_t err = vkllm_context_new(0, &context);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at creating context: %s", vkllm_err_s(err));
        return -1;
    }

    err = vkllm_models_llama2_load(context, &model, argv[1]);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at loading weights: %s", vkllm_err_s(err));
        goto cleanup_context;
    }

    struct vkllm_tensor *input_toks = NULL;
    uint32_t input_shapes[] = {1, 1, 1, 8};

    _CHECK_JUMP(vkllm_tensor_new(context, "input_toks", input_shapes, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0, NULL,
                                 0, true, &input_toks),
                err, cleanup_context);

    _CHECK_JUMP(vkllm_models_llama2_build_model(context, &model, input_toks), err, cleanup_input);
    _CHECK_JUMP(vkllm_models_llama2_free(context, &model), err, cleanup_input);
    return VKLLM_ERR_OK;

cleanup_input:
    vkllm_tensor_free(context, input_toks);
cleanup_context:
    vkllm_context_free(context);
    return err;
}
