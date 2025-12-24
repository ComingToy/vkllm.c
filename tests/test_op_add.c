#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_common.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_add.h"
#include "src/vkllm_tensor.h"
#include <float.h>
#include <stdio.h>

static void random_buf(void *a, const size_t n, vkllm_dtype_t dtype)
{
    if (dtype == vkllm_dtype_float16)
    {
        _Float16 *p = (_Float16 *)a;

        for (size_t i = 0; i < n; ++i)
        {
            p[i] = (_Float16)(10.0 * (rand() % 100) / 100.0);
        }

        return;
    }

    float *p = (float *)a;
    for (size_t i = 0; i < n; ++i)
    {
        p[i] = 10.0 * (rand() % 100) / 100.0;
    }
}

static void add_buf(const void *a, const void *b, void *c, size_t n, vkllm_dtype_t dtype)
{
    if (dtype == vkllm_dtype_float16)
    {
        const _Float16 *p0 = a, *p1 = b;
        float *p2 = c;

        for (size_t i = 0; i < n; ++i)
        {
            p2[i] = (float)(p0[i] + p1[i]);
        }

        return;
    }

    const float *p0 = a, *p1 = b;
    float *p2 = c;
    for (uint32_t i = 0; i < n; ++i)
    {
        p2[i] = p0[i] + p1[i];
    }
}

static float compare_buf(const void *lhs, const void *rhs, uint32_t shapes[4], uint32_t strides[4], uint32_t bytes,
                         vkllm_dtype_t dtype)
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
    const _Float16 *lhs_fp16 = lhs;
    const _Float16 *rhs_fp16 = rhs;

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
                        err = err + alpha * (float)((lhs_fp16[i] - rhs_fp16[i]) * (lhs_fp16[i] - rhs_fp16[i]));
                        continue;
                    }

                    err = err + alpha * (lhs_fp32[i] - rhs_fp32[i]) * (lhs_fp32[i] - rhs_fp32[i]);
                }
            }
        }
    }

    return err;
}

static void print_n(const char *prefix, const float *buf, const size_t n)
{
    fprintf(stderr, "%s: ", prefix);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(stderr, "%f ", buf[i]);
    }
    fprintf(stderr, "\n");
}

static struct
{
    uint32_t shapes[4];
    vkllm_dtype_t dtype;
} tests[] = {
    {{3, 4, 5, 99}, vkllm_dtype_float32},     {{3, 4, 5, 256}, vkllm_dtype_float32},
    {{13, 54, 42, 128}, vkllm_dtype_float32}, {{51, 33, 90, 31}, vkllm_dtype_float32},

    {{3, 4, 5, 99}, vkllm_dtype_float16},     {{3, 4, 5, 256}, vkllm_dtype_float16},
    {{13, 54, 42, 128}, vkllm_dtype_float16}, {{51, 33, 90, 31}, vkllm_dtype_float16},
};

START_TEST(test_op_add)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *in0, *in1, *out0;
    ck_assert_int_eq(vkllm_tensor_new(context, "in0", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL,
                                      0, false, &in0),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_new(context, "in1", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL,
                                      0, false, &in1),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *srcs[] = {in0, in1};
    ck_assert_int_eq(vkllm_tensor_new(context, "out0", tests[_i].shapes, vkllm_dtype_float32, VKLLM_OP_ADD, srcs, 2,
                                      NULL, 0, true, &out0),
                     VKLLM_ERR_OK);

    struct vkllm_dtype_info dtype_info;
    vkllm_get_dtype_info(tests[_i].dtype, &dtype_info);

    size_t n = in0->bytes / dtype_info.bytes;
    struct vkllm_array_f32 *buf, *buf1, *buf2;
    ck_assert_int_eq(vkllm_array_f32_new(&buf, n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_array_f32_new(&buf1, n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_array_f32_new(&buf2, out0->bytes / sizeof(float)), VKLLM_ERR_OK);

    random_buf(buf->data, n, tests[_i].dtype);
    random_buf(buf1->data, n, tests[_i].dtype);

    add_buf(buf->data, buf1->data, buf2->data, n, tests[_i].dtype);

    struct vkllm_array_f32 *buf3;
    vkllm_array_f32_new(&buf3, n);

    size_t bytes = dtype_info.bytes * buf->alloc_n;
    ck_assert_int_eq(bytes, in0->bytes);
    ck_assert_int_eq(bytes, in1->bytes);
    // ck_assert_int_eq(bytes, out0->bytes);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, in0, (const uint8_t *)buf->data, bytes), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, in1, (const uint8_t *)buf1->data, bytes), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_add(context, commands, out0), VKLLM_ERR_OK);
    // ck_assert_int_eq(vkllm_commands_download(context, commands, out0, (uint8_t *)buf3->data, out0->bytes),
    //                  VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_invalid_cache(context, out0), VKLLM_ERR_OK);

    const float *p = out0->data.host;
    // print_n("buf2", buf2->data, 64);
    // print_n("out0", p, 64);
    ck_assert_float_le(compare_buf(buf2->data, p, out0->shapes, out0->strides, out0->bytes, vkllm_dtype_float32), 1e-5);

    vkllm_tensor_free(context, in0);
    vkllm_tensor_free(context, in1);
    vkllm_tensor_free(context, out0);
    vkllm_array_f32_free(buf);
    vkllm_array_f32_free(buf1);
    vkllm_array_f32_free(buf2);
    vkllm_array_f32_free(buf3);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_add_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase_f32 = NULL, *tcase_f16 = NULL;
    suite = suite_create("vkllm_op_add");
    tcase_f32 = tcase_create("vkllm_op_add_f32");
    tcase_f16 = tcase_create("vkllm_op_add_f16");

    tcase_add_loop_test(tcase_f32, test_op_add, 0, 4);
    tcase_add_loop_test(tcase_f16, test_op_add, 4, 8);
    tcase_set_timeout(tcase_f32, 60.0);
    tcase_set_timeout(tcase_f16, 60.0);
    suite_add_tcase(suite, tcase_f32);
    suite_add_tcase(suite, tcase_f16);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_add_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
