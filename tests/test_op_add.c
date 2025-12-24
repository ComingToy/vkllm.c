#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_common.h"
#include "src/vkllm_context.h"
#include "src/vkllm_op_add.h"
#include "src/vkllm_tensor.h"
#include <stdio.h>

static void random_buf(float *a, const size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = 10.0 * (rand() % 100) / 100.0;
    }
}

static void add_buf(const float *a, const float *b, float *c, size_t n)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

static float compare_buf(const float *lhs, const float *rhs, uint32_t shapes[4], uint32_t strides[4], uint32_t bytes)
{
    uint32_t n = _MUL4(shapes);
    float alpha = 1.0 / n;
    float err = .0;

    // fprintf(stderr, "alpha: %f, bytes: %u, n: %u, en: %zu\n", alpha, bytes, n, bytes / sizeof(float));

    uint32_t es[4] = {strides[0] / sizeof(float), strides[1] / sizeof(float), strides[2] / sizeof(float),
                      strides[3] / sizeof(float)};

    // fprintf(stderr, "shapes = [%u, %u, %u, %u], strides = [%u, %u, %u, %u], es = [%u, %u, %u, %u]\n", shapes[0],
    //         shapes[1], shapes[2], shapes[3], strides[0], strides[1], strides[2], strides[3], es[0], es[1], es[2],
    //         es[3]);

    for (uint32_t b = 0; b < shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < shapes[2]; ++h)
            {
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t i = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                    err = err + alpha * (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
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

static uint32_t shapes[][4] = {
    {3, 4, 5, 99},
    {3, 4, 5, 256},
    {13, 54, 42, 128},
    {51, 33, 90, 31},
};

START_TEST(test_op_add_f32)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *in0, *in1, *out0;
    ck_assert_int_eq(
        vkllm_tensor_new(context, "in0", shapes[_i], vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, &in0),
        VKLLM_ERR_OK);
    ck_assert_int_eq(
        vkllm_tensor_new(context, "in1", shapes[_i], vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, &in1),
        VKLLM_ERR_OK);

    struct vkllm_tensor *srcs[] = {in0, in1};
    ck_assert_int_eq(
        vkllm_tensor_new(context, "out0", shapes[_i], vkllm_dtype_float32, VKLLM_OP_ADD, srcs, 2, NULL, 0, true, &out0),
        VKLLM_ERR_OK);

    size_t n = in0->bytes / sizeof(float);
    struct vkllm_array_f32 *buf, *buf1, *buf2;
    ck_assert_int_eq(vkllm_array_f32_new(&buf, n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_array_f32_new(&buf1, n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_array_f32_new(&buf2, n), VKLLM_ERR_OK);

    random_buf(buf->data, n);
    random_buf(buf1->data, n);

    add_buf(buf->data, buf1->data, buf2->data, n);

    struct vkllm_array_f32 *buf3;
    vkllm_array_f32_new(&buf3, n);

    size_t bytes = sizeof(float) * buf->alloc_n;
    ck_assert_int_eq(bytes, in0->bytes);
    ck_assert_int_eq(bytes, in1->bytes);
    ck_assert_int_eq(bytes, out0->bytes);

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
    // print_n("buf3", p, 64);
    ck_assert_float_eq(compare_buf(buf2->data, p, out0->shapes, out0->strides, out0->bytes), 0);
    // ck_assert_float_eq(compare_buf(buf2->data, buf3->data, n), 0);

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
    TCase *core = NULL;
    suite = suite_create("vkllm_op_add");
    core = tcase_create("vkllm_op_add_f32");

    tcase_add_loop_test(core, test_op_add_f32, 0, 4);
    tcase_set_timeout(core, 60.0);
    suite_add_tcase(suite, core);
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
