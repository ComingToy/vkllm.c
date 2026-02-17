#ifndef __VKLLM_TEST_COMMON_H__
#define __VKLLM_TEST_COMMON_H__

#include "src/vkllm_dtypes.h"
#include "src/vkllm_common.h"
#include <stdlib.h>
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

static inline void random_buf(void *a, const size_t n, vkllm_dtype_t dtype)
{
    if (dtype == vkllm_dtype_float16)
    {
        vkllm_fp16_pack *p = (vkllm_fp16_pack *)a;

        for (size_t i = 0; i < n; ++i)
        {
            p[i] = vkllm_fp32_to_fp16(10.0 * (rand() % 100) / 100.0);
        }

        return;
    }

    float *p = (float *)a;
    for (size_t i = 0; i < n; ++i)
    {
        p[i] = 10.0 * (rand() % 100) / 100.0;
    }
}

static inline float compare_buf(const void *lhs, const void *rhs, uint32_t shapes[4], uint32_t strides[4],
                                uint32_t bytes, vkllm_dtype_t dtype)
{
    uint32_t n = _MUL4(shapes);
    float alpha = 1.0 / n;
    float err = .0;

    // fprintf(stderr, "alpha: %f, bytes: %u, n: %u, en: %zu\n", alpha, bytes, n, bytes / sizeof(float));

    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);

    uint32_t es[4] = {strides[0] / info.bytes, strides[1] / info.bytes, strides[2] / info.bytes,
                      strides[3] / info.bytes};

    // fprintf(stderr, "shapes = [%u, %u, %u, %u], strides = [%u, %u, %u, %u], es = [%u, %u, %u, %u]\n", shapes[0],
    //         shapes[1], shapes[2], shapes[3], strides[0], strides[1], strides[2], strides[3], es[0], es[1], es[2],
    //         es[3]);

    const float *lhs_fp32 = lhs;
    const float *rhs_fp32 = rhs;
    const vkllm_fp16_pack *lhs_fp16 = lhs;
    const vkllm_fp16_pack *rhs_fp16 = rhs;

    for (uint32_t b = 0; b < shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < shapes[2]; ++h)
            {
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t i = b * es[0] + c * es[1] + h * es[2] + w * es[3];

                    if (dtype == vkllm_dtype_float16)
                    {
                        float v0 = vkllm_fp16_to_fp32(lhs_fp16[i]);
                        float v1 = vkllm_fp16_to_fp32(rhs_fp16[i]);
                        err = err + alpha * (v0 - v1) * (v0 - v1);
                        continue;
                    }

                    err = err + alpha * (lhs_fp32[i] - rhs_fp32[i]) * (lhs_fp32[i] - rhs_fp32[i]);
                }
            }
        }
    }

    return err;
}

static inline void print_n(const char *prefix, const float *buf, const size_t n)
{
    fprintf(stderr, "%s: ", prefix);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(stderr, "%f ", buf[i]);
    }
    fprintf(stderr, "\n");
}

#endif
