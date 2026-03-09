#include "check.h"
#include "src/core/vkllm_array.h"
#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_op_arg_max.h"
#include "src/core/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <math.h>
#include <string.h>

static void arg_max_op_host(const void *input, uint32_t *output, const uint32_t shapes[4], const uint32_t strides[4],
                            const uint32_t out_shapes[4], const uint32_t out_strides[4], vkllm_dtype_t dtype)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);
    uint32_t dsize = info.bytes;

    const float *input_fp32 = (const float *)input;
    const vkllm_fp16_pack *input_fp16 = (const vkllm_fp16_pack *)input;

    uint32_t es[4] = {strides[0] / dsize, strides[1] / dsize, strides[2] / dsize, strides[3] / dsize};
    uint32_t out_es[4] = {out_strides[0] / sizeof(uint32_t), out_strides[1] / sizeof(uint32_t),
                          out_strides[2] / sizeof(uint32_t), out_strides[3] / sizeof(uint32_t)};

    uint32_t B = shapes[0];
    uint32_t C = shapes[1];
    uint32_t H = shapes[2];
    uint32_t W = shapes[3];

    for (uint32_t b = 0; b < B; ++b)
    {
        for (uint32_t c = 0; c < C; ++c)
        {
            for (uint32_t h = 0; h < H; ++h)
            {
                uint32_t out_idx = b * out_es[0] + c * out_es[1] + h * out_es[2];

                uint32_t max_idx = 0;
                float max_val = -INFINITY;

                for (uint32_t w = 0; w < W; ++w)
                {
                    uint32_t in_idx = b * es[0] + c * es[1] + h * es[2] + w * es[3];

                    float val;
                    if (dtype == vkllm_dtype_float32)
                    {
                        val = input_fp32[in_idx];
                    }
                    else if (dtype == vkllm_dtype_float16)
                    {
                        val = vkllm_fp16_to_fp32(input_fp16[in_idx]);
                    }
                    else
                    {
                        val = -INFINITY;
                    }

                    if (val > max_val)
                    {
                        max_val = val;
                        max_idx = w;
                    }
                }

                output[out_idx] = max_idx;
            }
        }
    }
}

static struct
{
    uint32_t shapes[4];
    vkllm_dtype_t dtype;
} tests[] = {
    {{1, 1, 1, 100}, vkllm_dtype_float32},    {{1, 1, 1, 100}, vkllm_dtype_float16},
    {{1, 1, 10, 128}, vkllm_dtype_float32},   {{1, 1, 10, 128}, vkllm_dtype_float16},
    // {{2, 4, 8, 64}, vkllm_dtype_float32},     {{2, 4, 8, 64}, vkllm_dtype_float16},
    // {{3, 5, 7, 256}, vkllm_dtype_float32},    {{1, 12, 16, 512}, vkllm_dtype_float16},
    // {{8, 16, 32, 1024}, vkllm_dtype_float32}, {{1, 1, 1, 10}, vkllm_dtype_float32},
    // {{4, 8, 16, 32}, vkllm_dtype_float16},
};

START_TEST(test_op_arg_max)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *input;
    ck_assert_int_eq(vkllm_tensor_new(context, "input", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL,
                                      0, true, &input),
                     VKLLM_ERR_OK);

    uint32_t output_shapes[4];
    memcpy(output_shapes, tests[_i].shapes, sizeof(output_shapes));
    output_shapes[3] = 1;

    struct vkllm_tensor *srcs[] = {input};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", output_shapes, vkllm_dtype_uint32, VKLLM_OP_ARG_MAX, srcs, 1,
                                      NULL, 0, true, &output),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *input_host = NULL, *output_host = NULL;

    vkllm_array_u8_new(&input_host, input->bytes);
    vkllm_array_u8_new(&output_host, output->bytes);

    memset(output_host->data, 0, output_host->alloc_n);

    random_tensor(input_host->data, input->shapes, input->strides, input->dtype, -10.0, 10.0);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input, input_host->data, input_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_arg_max_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_arg_max_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_arg_max_post_run(context, commands, output), VKLLM_ERR_OK);

    arg_max_op_host(input_host->data, (uint32_t *)output_host->data, input->shapes, input->strides, output->shapes,
                    output->strides, input->dtype);

    const uint32_t *gpu_output = (const uint32_t *)output->data.host;
    const uint32_t *host_output = (const uint32_t *)output_host->data;

    uint32_t n = output->bytes / sizeof(uint32_t);
    uint32_t n_errors = 0;

    for (uint32_t i = 0; i < n; ++i)
    {
        if (gpu_output[i] != host_output[i])
        {
            n_errors++;
            if (n_errors <= 5)
            {
                log_error("Mismatch at index %u: gpu=%u, host=%u", i, gpu_output[i], host_output[i]);
            }
        }
    }

    for (uint32_t i = 0; i < n; ++i)
    {
        if (gpu_output[i] != host_output[i])
        {
            n_errors++;
            if (n_errors <= 5)
            {
                log_error("Mismatch at index %u: gpu=%u, host=%u", i, gpu_output[i], host_output[i]);
            }
        }
    }

    if (n_errors > 0)
    {
        log_error("Test case %d failed: %u/%u mismatches", _i, n_errors, n);
        log_error("Shapes: [%u, %u, %u, %u], dtype: %s", tests[_i].shapes[0], tests[_i].shapes[1], tests[_i].shapes[2],
                  tests[_i].shapes[3], vkllm_dtype_s(tests[_i].dtype));
    }

    ck_assert_int_eq(n_errors, 0);

    vkllm_tensor_free(context, input);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(input_host);
    vkllm_array_u8_free(output_host);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_arg_max_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase;
    suite = suite_create("vkllm_op_arg_max");
    tcase = tcase_create("vkllm_op_arg_max");

    tcase_add_loop_test(tcase, test_op_arg_max, 0, sizeof(tests) / sizeof(tests[0]));
    tcase_set_timeout(tcase, 60.0);
    suite_add_tcase(suite, tcase);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_arg_max_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
