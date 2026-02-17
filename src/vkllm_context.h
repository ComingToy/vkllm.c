#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#include <zlog.h>

#include "vkllm_errors.h"

struct vkllm_context {
    zlog_category_t* zlog_c;
};

extern vkllm_err_t new_vkllm_context(zlog_category_t* zlog_c,
				     struct vkllm_context** context);

#endif
