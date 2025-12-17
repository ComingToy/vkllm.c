#include "vkllm_op_embedding.h"
#include "src/vkllm_common.h"
#include "vkllm_commands.h"
#include "vkllm_context.h"
#include "vkllm_tensor.h"

vkllm_err_t vkllm_op_embedding(struct vkllm_context *context, struct vkllm_commands *commands,
                               struct vkllm_tensor *tensor)
{
    _CHECK_ARGS(context && commands && tensor);
    return VKLLM_ERR_OK;
}
