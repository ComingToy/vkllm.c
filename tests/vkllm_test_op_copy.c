#include "check.h"
#include "src/core/vkllm_array.h"
#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_op_copy.h"
#include "src/core/vkllm_pipeline.h"
#include "src/core/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <stdio.h>
#include <string.h>

static struct vkllm_context *g_context = NULL;

static void copy_op_host(const void *input, void *output, const uint32_t shapes[4], const uint32_t strides[4],
                         vkllm_dtype_t dtype)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);

    uint32_t es[4] = {strides[0] / info.bytes, strides[1] / info.bytes, strides[2] / info.bytes,
                      strides[3] / info.bytes};

    if (dtype == vkllm_dtype_float16)
    {
        const vkllm_fp16_pack *input_fp16 = (const vkllm_fp16_pack *)input;
        vkllm_fp16_pack *output_fp16 = (vkllm_fp16_pack *)output;
        for (uint32_t b = 0; b < shapes[0]; ++b)
        {
            for (uint32_t c = 0; c < shapes[1]; ++c)
            {
                for (uint32_t h = 0; h < shapes[2]; ++h)
                {
                    for (uint32_t w = 0; w < shapes[3]; ++w)
                    {
                        uint32_t idx_in = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                        output_fp16[idx_in] = input_fp16[idx_in];
                    }
                }
            }
        }
    }
    else
    {
        const float *input_fp32 = (const float *)input;
        float *output_fp32 = (float *)output;
        for (uint32_t b = 0; b < shapes[0]; ++b)
        {
            for (uint32_t c = 0; c < shapes[1]; ++c)
            {
                for (uint32_t h = 0; h < shapes[2]; ++h)
                {
                    for (uint32_t w = 0; w < shapes[3]; ++w)
                    {
                        uint32_t idx_in = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                        output_fp32[idx_in] = input_fp32[idx_in];
                    }
                }
            }
        }
    }
}

static struct
{
    uint32_t shapes[4];
    vkllm_dtype_t dtype;
} tests[] = {
    {{1, 1, 128, 256}, vkllm_dtype_float16},  {{2, 4, 64, 128}, vkllm_dtype_float16},
    {{1, 1, 128, 256}, vkllm_dtype_float32},  {{2, 4, 64, 128}, vkllm_dtype_float32},
    {{4, 8, 32, 64}, vkllm_dtype_float32},

    {{8, 16, 128, 256}, vkllm_dtype_float16}, {{8, 16, 128, 256}, vkllm_dtype_float32},

    {{1, 1, 1, 1}, vkllm_dtype_float16},      {{1, 1, 1, 1}, vkllm_dtype_float32},
};

START_TEST(test_op_copy)
{
    struct vkllm_context *context = g_context;
    ck_assert_ptr_nonnull(context);

    struct vkllm_commands *commands;
    vkllm_err_t err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    uint32_t B = tests[_i].shapes[0];
    uint32_t C = tests[_i].shapes[1];
    uint32_t H = tests[_i].shapes[2];
    uint32_t W = tests[_i].shapes[3];

    struct vkllm_tensor *input;
    ck_assert_int_eq(vkllm_tensor_new(context, "input", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL,
                                      0, false, &input),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *output;
    struct vkllm_tensor *srcs[] = {input};
    ck_assert_int_eq(vkllm_tensor_new(context, "output", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_COPY, srcs, 1,
                                      NULL, 0, true, &output),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *buf_input = NULL, *buf_expected = NULL;

    vkllm_array_u8_new(&buf_input, input->bytes);
    vkllm_array_u8_new(&buf_expected, output->bytes);

    memset(buf_input->data, 0, buf_input->alloc_n);
    memset(buf_expected->data, 0, buf_expected->alloc_n);

    random_tensor(buf_input->data, input->shapes, input->strides, input->dtype, 0.0, 1.0);

    copy_op_host(buf_input->data, buf_expected->data, tests[_i].shapes, input->strides, tests[_i].dtype);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input, buf_input->data, buf_input->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_copy_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_copy_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_copy_post_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);

    uint64_t time_cost = 0;
    ck_assert_int_eq(vkllm_pipeline_query_exec_time(context, output->pipeline, &time_cost), VKLLM_ERR_OK);

    log_info("test %d copy_op [%u,%u,%u,%u], dtype=%s: avg time cost: %lu micro secs", _i, B, C, H, W,
             vkllm_dtype_s(tests[_i].dtype), time_cost / 1000);

    const void *gpu_output = output->data.host;

    float tolerance = (tests[_i].dtype == vkllm_dtype_float16) ? 1e-2 : 1e-4;
    char test_case_name[64];
    snprintf(test_case_name, sizeof(test_case_name), "test_op_copy_%d", _i);
    float error = compare_buf(buf_expected->data, gpu_output, output->shapes, output->strides, output->bytes,
                              output->dtype, test_case_name);
    ck_assert_float_le(error, tolerance);

    vkllm_tensor_free(context, input);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(buf_input);
    vkllm_array_u8_free(buf_expected);
    vkllm_commands_free(context, commands);
}

END_TEST;

static void setup_global_context(void)
{
    vkllm_err_t err = vkllm_context_new(0, &g_context);
    if (err != VKLLM_ERR_OK)
    {
        fprintf(stderr, "Failed to create global context: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

static void teardown_global_context(void)
{
    if (g_context != NULL)
    {
        vkllm_context_free(g_context);
        g_context = NULL;
    }
}

Suite *vkllm_op_copy_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase = NULL;
    suite = suite_create("vkllm_op_copy");
    tcase = tcase_create("vkllm_op_copy");

    tcase_add_unchecked_fixture(tcase, setup_global_context, teardown_global_context);

    tcase_add_loop_test(tcase, test_op_copy, 0, sizeof(tests) / sizeof(tests[0]));
    tcase_set_timeout(tcase, 120.0);
    suite_add_tcase(suite, tcase);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_copy_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}