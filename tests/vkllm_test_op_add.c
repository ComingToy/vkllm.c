#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_bin.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <stdio.h>

static inline void add_buf(const void *a, const void *b, void *c, size_t n, vkllm_dtype_t dtype)
{
    if (dtype == vkllm_dtype_float16)
    {
        const vkllm_fp16_pack *p0 = a, *p1 = b;
        float *p2 = c;

        for (size_t i = 0; i < n; ++i)
        {
            p2[i] = vkllm_fp16_to_fp32(p0[i]) + vkllm_fp16_to_fp32(p1[i]);
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

static struct
{
    uint32_t shapes[4];
    vkllm_dtype_t dtype;
} tests[] = {
    {{3, 4, 5, 99}, vkllm_dtype_float32},      {{3, 4, 5, 256}, vkllm_dtype_float32},
    {{13, 54, 42, 128}, vkllm_dtype_float32},  {{51, 33, 90, 31}, vkllm_dtype_float32},
    {{10, 10, 10, 1024}, vkllm_dtype_float32}, {{10, 10, 10, 1013}, vkllm_dtype_float32},

    {{3, 4, 5, 99}, vkllm_dtype_float16},      {{3, 4, 5, 256}, vkllm_dtype_float16},
    {{13, 54, 42, 128}, vkllm_dtype_float16},  {{51, 33, 90, 31}, vkllm_dtype_float16},
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
    int32_t op = 0;
    ck_assert_int_eq(vkllm_tensor_new(context, "out0", tests[_i].shapes, vkllm_dtype_float32, VKLLM_OP_BIN, srcs, 2,
                                      (uint8_t *)&op, sizeof(op), true, &out0),
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
    ck_assert_int_eq(vkllm_op_bin_init(context, commands, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_bin_run(context, commands, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_bin_post_run(context, commands, out0), VKLLM_ERR_OK);
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

    tcase_add_loop_test(tcase_f32, test_op_add, 0, 6);
    tcase_add_loop_test(tcase_f16, test_op_add, 6, 10);
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
