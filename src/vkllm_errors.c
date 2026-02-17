#include "vkllm_errors.h"

const char* vkllm_err_s(vkllm_err_t err) {
#define _VKLLM_ERR_OP(_err) #_err,
    static const char* const _errs_table[] = {
#include "vkllm_errors.inc"
    };
#undef _VKLLM_ERR_OP

    return _errs_table[(int)err - 1];
}

