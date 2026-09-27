#ifndef __VKLLM_ERRORS_H__
#define __VKLLM_ERRORS_H__

#ifdef __cplusplus
extern "C" {
#endif
#define _VKLLM_ERR_OP(_err) _err,

// VKLLM_ERR_OK is 0 (C convention: zero = success), so process exit codes
// and boolean-ish checks behave as expected.
typedef enum
{
#include "vkllm_errors.inc"
} vkllm_err_t;

#undef _VKLLM_ERR_OP

extern const char *vkllm_err_s(vkllm_err_t err);

#ifdef __cplusplus
}
#endif
#endif
