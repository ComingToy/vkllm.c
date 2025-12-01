#ifndef __VKLLM_ERRORS_H__
#define __VKLLM_ERRORS_H__

#define _VKLLM_ERR_OP(_err) _err,

typedef enum {
    VKLLM_ERR_START = 0,
#include "vkllm_errors.inc"
} vkllm_err_t;

#undef _VKLLM_ERR_OP

extern const char* vkllm_err_s(vkllm_err_t err);

#endif
