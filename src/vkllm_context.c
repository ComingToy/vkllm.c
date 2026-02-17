#include "vkllm_context.h"

#include "vkllm_common.h"

vkllm_err_t new_vkllm_context(zlog_category_t* zlog_c,
                              struct vkllm_context** context) {
    _NEW_AND_CHECK(*context, struct vkllm_context);

    struct vkllm_context* p = *context;
    p->zlog_c = zlog_c;

    return VKLLM_ERR_OK;
}
