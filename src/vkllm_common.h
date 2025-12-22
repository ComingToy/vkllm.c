#ifndef __VKLLM_COMMON_H__
#define __VKLLM_COMMON_H__

#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#define _CHECK_JUMP(_fn, _ret, _label)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        _ret = (_fn);                                                                                                  \
        if (_ret != VKLLM_ERR_OK)                                                                                      \
        {                                                                                                              \
            goto _label;                                                                                               \
        }                                                                                                              \
    } while (0)

#define _CHECK_VK(__fn)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        VkResult __err = (__fn);                                                                                       \
        if (__err != VK_SUCCESS)                                                                                       \
        {                                                                                                              \
            log_error(#__fn " failed: %d", (int)__err);                                                                \
            return VKLLM_ERR_VULKAN;                                                                                   \
        }                                                                                                              \
    } while (0)

#define _CHECK_VK_JUMP(__fn, __ret, __label)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        VkResult __err = (__fn);                                                                                       \
        if (__err != VK_SUCCESS)                                                                                       \
        {                                                                                                              \
            log_error(#__fn " failed: %d", (int)__err);                                                                \
            __ret = VKLLM_ERR_VULKAN;                                                                                  \
            goto __label;                                                                                              \
        }                                                                                                              \
    } while (0)

#define BOOL_S(_cond) (!!(_cond) ? "false" : "true")
#define _CHECK_ARGS(_cond)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(_cond))                                                                                                  \
        {                                                                                                              \
            log_error(#_cond " is false");                                                                             \
            return VKLLM_ERR_ARGS;                                                                                     \
        }                                                                                                              \
    } while (0)

#define _DIV4_S(__vec4, __den, __out)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        (__out)[0] = (__vec4)[0] / (__den);                                                                            \
        (__out)[1] = (__vec4)[1] / (__den);                                                                            \
        (__out)[2] = (__vec4)[2] / (__den);                                                                            \
        (__out)[3] = (__vec4)[3] / (__den);                                                                            \
    } while (0)

#define _MUL4(__vec4) ((__vec4)[0] * (__vec4)[1] * (__vec4)[2] * (__vec4)[3])

#define _ARRAY_SIZE(_arr) (sizeof(_arr)/sizeof((_arr)[0]))

#define __PACKED__ __attribute__((packed))
// configs

#define VKLLM_MAX_PHY_DEVS 16
#define VKLLM_MAX_SRCS 4

#ifdef __cplusplus
}
#endif
#endif
