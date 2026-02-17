#ifndef __VKLLM_COMMON_H__
#define __VKLLM_COMMON_H__

#include <stddef.h>
#include <stdlib.h>

#include "vkllm_errors.h"

#define _NEW_AND_CHECK(_p, _type)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        (_p) = (_type *)malloc(sizeof(_type));                                                                         \
        if (!(_p))                                                                                                     \
            return VKLLM_ERR_ALLOC;                                                                                    \
    } while (0)

#define _NEW_N_AND_CHECK(_p, _type, _n)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        (_p) = (_type *)malloc(sizeof(_type) * (_n));                                                                  \
        if (!(_p))                                                                                                     \
            return VKLLM_ERR_ALLOC;                                                                                    \
    } while (0)

#define _CHECK(fn)                                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        vkllm_err_t __r = (fn);                                                                                        \
        if (__r != VKLLM_ERR_OK)                                                                                       \
            return __r;                                                                                                \
    } while (0)

#define BOOL_S(_cond) (!!(_cond) ? "false" : "true")
// configs

#define VKLLM_MAX_PHY_DEVS 16
#define VKLLM_MAX_SRCS 4
#endif
