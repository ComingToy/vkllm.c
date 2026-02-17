#ifndef __VKLLM_ERRORS_H__
#define __VKLLM_ERRORS_H__

#define _VKLLM_ERR_OP(_err) _err,

typedef enum {
    VKLLM_ERR_START = 0,
#include "vkllm_errors.inc"
} vkllm_err_t;

#undef _VKLLM_ERR_OP

inline const char* vkllm_err_s(vkllm_err_t err) {
#define _VKLLM_ERR_OP(_err) #_err,
    static const char* const _errs_table[] = {
#include "vkllm_errors.inc"
    };
#undef _VKLLM_ERR_OP

    return _errs_table[err];
}

#endif
