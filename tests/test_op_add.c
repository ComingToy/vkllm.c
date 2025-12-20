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

static float compare_buf(const float *a, const float *b, size_t n)
{
    float w = 1.0 / n;
    float err = .0;
    for (uint32_t i = 0; i < n; ++i)
    {
        err = err + w * (a[i] - b[i]) * (a[i] - b[i]);
    }

    return err;
}

static uint32_t shapes[][4] = {
    {3, 4, 5, 128},
    {3, 4, 5, 99},
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

    size_t n = _MUL4(shapes[_i]);

    struct vkllm_array_f32 *buf, *buf1, *buf2;
    ck_assert_int_eq(vkllm_array_f32_new(&buf, n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_array_f32_new(&buf1, n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_array_f32_new(&buf2, n), VKLLM_ERR_OK);

    random_buf(buf->data, n);
    random_buf(buf1->data, n);
    add_buf(buf->data, buf1->data, buf2->data, n);

    struct vkllm_tensor *in0, *in1, *out0;
    vkllm_tensor_new(context, "in0", shapes[_i], vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, &in0);
    vkllm_tensor_new(context, "in1", shapes[_i], vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, &in1);

    struct vkllm_tensor *srcs[] = {in0, in1};
    vkllm_tensor_new(context, "out0", shapes[_i], vkllm_dtype_float32, VKLLM_OP_ADD, srcs, 2, NULL, 0, true, &out0);

    size_t bytes = sizeof(float) * buf->alloc_n;
    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, in0, (const uint8_t *)buf->data, bytes), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, in1, (const uint8_t *)buf1->data, bytes), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_add(context, commands, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_invalid_cache(context, out0), VKLLM_ERR_OK);

    const float *p = out0->data.host;
    ck_assert_float_eq(compare_buf(buf2->data, p, n), 0);

    vkllm_tensor_free(context, in0);
    vkllm_tensor_free(context, in1);
    vkllm_tensor_free(context, out0);
    vkllm_array_f32_free(buf);
    vkllm_array_f32_free(buf1);
    vkllm_array_f32_free(buf2);
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

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
