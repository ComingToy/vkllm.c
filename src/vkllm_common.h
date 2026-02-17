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

#define BOOL_S(_cond) (!!(_cond) ? "true" : "false")
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

#define _ARRAY_SIZE(_arr) (sizeof(_arr) / sizeof((_arr)[0]))

#define __PACKED__ __attribute__((packed))

#define __UNUSED(__x) ((void)__x)

// configs

#define VKLLM_MAX_PHY_DEVS 16
#define VKLLM_MAX_SRCS 4

typedef struct fp16_pack
{
    unsigned short frac : 10;
    unsigned char exp : 5;
    unsigned char sign : 1;
} __attribute__((packed)) vkllm_fp16_pack;

struct fp32_pack
{
    unsigned int frac : 23;
    unsigned char exp : 8;
    unsigned char sign : 1;
} __attribute__((packed));

static inline float vkllm_fp16_to_fp32(vkllm_fp16_pack data)
{
    float f;
    struct fp32_pack *fp32 = (struct fp32_pack *)&f;
    struct fp16_pack *fp16 = &data;

    int exp = fp16->exp;

    if (exp == 31 && fp16->frac != 0)
    {
        // return __builtin_inf()-__builtin_inf();
        fp32->sign = fp16->sign;
        fp32->exp = 255;
        fp32->frac = 1;

        return f;
    }

    if (exp == 31)
        exp = 255;
    if (exp == 0)
        exp = 0;
    else
        exp = (exp - 15) + 127;

    fp32->exp = exp;
    fp32->sign = fp16->sign;
    fp32->frac = ((int)fp16->frac) << 13;

    return f;
}

static inline vkllm_fp16_pack vkllm_fp32_to_fp16(float data)
{
    struct fp32_pack *fp32 = (struct fp32_pack *)&data;
    struct fp16_pack fp16;

    int exp = fp32->exp;

    if (fp32->exp == 255 && fp32->frac != 0)
    {
        // NaN
        fp16.exp = 31;
        fp16.frac = 1;
        fp16.sign = fp32->sign;

        return fp16;
    }

    if ((exp - 127) < -14)
        exp = 0;
    else if ((exp - 127) > 15)
        exp = 31;
    else
        exp = exp - 127 + 15;

    fp16.exp = exp;
    fp16.frac = fp32->frac >> 13;
    fp16.sign = fp32->sign;

    return fp16;
}

#ifdef __cplusplus
}
#endif
#endif
