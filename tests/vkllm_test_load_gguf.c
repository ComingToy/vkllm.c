#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_errors.h"
#include "src/models/vkllm_models_llama2.h"

int main(const int argc, const char *argv[])
{
    if (argc != 2)
    {
        log_error("usage: %s <path to gguf>", argv[0]);
        return -1;
    }

    struct vkllm_context *context = NULL;
    struct vkllm_models_llama2_weights model;

    vkllm_err_t err = vkllm_context_new(0, &context);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at creating context: %s", vkllm_err_s(err));
        return -1;
    }

    err = vkllm_models_llama2_load_weights(context, &model, argv[1]);
    if (err != VKLLM_ERR_OK)
    {
        log_error("failed at loading weights: %s", vkllm_err_s(err));
        goto cleanup;
    }

    _CHECK_JUMP(vkllm_models_llama2_free_weights(context, &model), err, cleanup);

cleanup:
    vkllm_context_free(context);
    return 0;
}
