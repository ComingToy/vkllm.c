#ifndef __VKLLM_TEST_COMMON_H__
#define __VKLLM_TEST_COMMON_H__

#include "src/core/vkllm_common.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_tensor.h"
#include "src/core/vkllm_context.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static inline void random_tensor(void *data, uint32_t shapes[4], uint32_t strides[4], vkllm_dtype_t dtype, float min, float max)
{

    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);
    uint32_t es[4] = {strides[0] / info.bytes, strides[1] / info.bytes, strides[2] / info.bytes,
                      strides[3] / info.bytes};

    vkllm_fp16_pack *fp16 = (vkllm_fp16_pack *)data;
    float *fp32 = (float *)data;
    uint32_t *u32 = (uint32_t *)data;

    for (uint32_t b = 0; b < shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < shapes[2]; ++h)
            {
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t i = b * es[0] + c * es[1] + h * es[2] + w * es[3];
					float val = (rand() % 1000) / 1000.0f;
					val = val * (max - min) + min;
                    if (dtype == vkllm_dtype_float16)
                    {
                        fp16[i] = vkllm_fp32_to_fp16(val);
                    }
                    else if (dtype == vkllm_dtype_float32)
                    {
                        fp32[i] = val;
                    }
                    else if (dtype == vkllm_dtype_uint32)
                    {
                        u32[i] = (uint32_t)((rand() % 10) / 10);
                    }
                }
            }
        }
    }
}

static inline uint32_t get_indice(uint32_t b, uint32_t c, uint32_t h, uint32_t w, const uint32_t strides[4],
                                  uint32_t dsize)
{
    uint32_t es[] = {strides[0] / dsize, strides[1] / dsize, strides[2] / dsize, strides[3] / dsize};
    uint32_t i = b * es[0] + c * es[1] + h * es[2] + w * es[3];
    return i;
}

#define _LOOP_SHAPE(shapes, _body)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        for (uint32_t _b = 0; _b < shapes[0]; ++_b)                                                                    \
        {                                                                                                              \
            for (uint32_t _c = 0; _c < shapes[1]; ++_c)                                                                \
            {                                                                                                          \
                for (uint32_t _h = 0; _h < shapes[2]; ++_h)                                                            \
                {                                                                                                      \
                    for (uint32_t _w = 0; _w < shapes[3]; ++_w)                                                        \
                    {                                                                                                  \
                        _body;                                                                                         \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

static inline float compare_buf(const void *lhs, const void *rhs, uint32_t shapes[4], uint32_t strides[4],
                                uint32_t bytes, vkllm_dtype_t dtype, const char* name)
{

    // fprintf(stderr, "alpha: %f, bytes: %u, n: %u, en: %zu\n", alpha, bytes, n, bytes / sizeof(float));

    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);

    uint32_t n = bytes / info.bytes;
    float alpha = 1.0 / _MUL4(shapes);
    float err = .0;

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
                    if (i > n)
                    {
                        log_error("index %u at (%u, %u, %u, %u) out of range %u", i, b, c, h, w, n);
                        continue;
                    }

                    if (dtype == vkllm_dtype_float16)
                    {
                        float v0 = vkllm_fp16_to_fp32(lhs_fp16[i]);
                        float v1 = vkllm_fp16_to_fp32(rhs_fp16[i]);
                        err = err + alpha * (v0 - v1) * (v0 - v1);

#if 0
                        if (fabsf(v0 - v1) > 1e-2 || isnan(err))
                        {
                            log_error("%s index %u at (%u, %u, %u, %u) err lhs %f rhs %f", name, i, b, c, h, w, v0, v1);
                            continue;
                        }
#endif

                        continue;
                    }

                    err = err + alpha * (lhs_fp32[i] - rhs_fp32[i]) * (lhs_fp32[i] - rhs_fp32[i]);
#if 0
                    if (fabsf(lhs_fp32[i] - rhs_fp32[i]) > 1e-3 || isnan(err))
                    {
                        log_error("index %u at (%u, %u, %u, %u) err lhs %f rhs %f", i, b, c, h, w, lhs_fp32[i],
                                  rhs_fp32[i]);
                        continue;
                    }
#endif
                }
            }
        }
    }

    return err;
}

static inline void print_n_f16(const char *prefix, const void *buf, const size_t n)
{
    fprintf(stderr, "%s\n", prefix);

    vkllm_fp16_pack *f16 = (vkllm_fp16_pack *)buf;
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(stderr, "%f ", vkllm_fp16_to_fp32(f16[i]));
    }
    fprintf(stderr, "\n");
}

static inline void print_n(const char *prefix, const float *buf, const size_t n)
{
    fprintf(stderr, "%s\n", prefix);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(stderr, "%f ", buf[i]);
    }
    fprintf(stderr, "\n");
}

static inline vkllm_err_t print_first_n(struct vkllm_context *context, struct vkllm_commands *commands,
                                 struct vkllm_tensor *tensor, uint32_t b, uint32_t c, uint32_t h, uint32_t n)
{
    if (!tensor->data.mapped)
    {
        return VKLLM_ERR_ARGS;
    }

    uint8_t *data = (uint8_t *)tensor->data.host;
    uint32_t count = n < tensor->shapes[3] ? n : tensor->shapes[3];

    fprintf(stderr, "%s [%u,%u,%u,:%u]: ", tensor->name, b, c, h, count);

    switch (tensor->dtype)
    {
    case vkllm_dtype_float32: {
        float *base = (float *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(float);
        uint32_t stride1 = tensor->strides[1] / sizeof(float);
        uint32_t stride2 = tensor->strides[2] / sizeof(float);
        uint32_t stride3 = tensor->strides[3] / sizeof(float);
        float *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%f ", p[w * stride3]);
        break;
    }
    case vkllm_dtype_float16: {
        vkllm_fp16_pack *base = (vkllm_fp16_pack *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(vkllm_fp16_pack);
        uint32_t stride1 = tensor->strides[1] / sizeof(vkllm_fp16_pack);
        uint32_t stride2 = tensor->strides[2] / sizeof(vkllm_fp16_pack);
        uint32_t stride3 = tensor->strides[3] / sizeof(vkllm_fp16_pack);
        vkllm_fp16_pack *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%f ", vkllm_fp16_to_fp32(p[w * stride3]));
        break;
    }
    case vkllm_dtype_int8: {
        int8_t *base = (int8_t *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(int8_t);
        uint32_t stride1 = tensor->strides[1] / sizeof(int8_t);
        uint32_t stride2 = tensor->strides[2] / sizeof(int8_t);
        uint32_t stride3 = tensor->strides[3] / sizeof(int8_t);
        int8_t *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%d ", p[w * stride3]);
        break;
    }
    case vkllm_dtype_uint32: {
        uint32_t *base = (uint32_t *)data;
        uint32_t stride0 = tensor->strides[0] / sizeof(uint32_t);
        uint32_t stride1 = tensor->strides[1] / sizeof(uint32_t);
        uint32_t stride2 = tensor->strides[2] / sizeof(uint32_t);
        uint32_t stride3 = tensor->strides[3] / sizeof(uint32_t);
        uint32_t *p = base + b * stride0 + c * stride1 + h * stride2;
        for (uint32_t w = 0; w < count; ++w)
            fprintf(stderr, "%u ", p[w * stride3]);
        break;
    }
    default:
        return VKLLM_ERR_ARGS;
    }

    fprintf(stderr, "\n");
    return VKLLM_ERR_OK;
}
#endif
