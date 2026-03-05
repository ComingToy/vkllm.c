#include "check.h"
#include "src/core/vkllm_array.h"
#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_op_update_rows.h"
#include "src/core/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <string.h>

static float compare_buf_offsets(const void *lhs, const void *rhs, uint32_t shapes[4], uint32_t strides[4],
                                 uint32_t offsets[4], uint32_t bytes, vkllm_dtype_t dtype, const char *name)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);

    uint32_t n = bytes / info.bytes;
    uint32_t count = 0;
    float err = .0f;

    uint32_t es[4] = {strides[0] / info.bytes, strides[1] / info.bytes, strides[2] / info.bytes,
                      strides[3] / info.bytes};

    const float *lhs_fp32 = lhs;
    const float *rhs_fp32 = rhs;
    const vkllm_fp16_pack *lhs_fp16 = lhs;
    const vkllm_fp16_pack *rhs_fp16 = rhs;

    for (uint32_t b = offsets[0]; b < shapes[0]; ++b)
    {
        for (uint32_t c = offsets[1]; c < shapes[1]; ++c)
        {
            for (uint32_t h = offsets[2]; h < shapes[2]; ++h)
            {
                for (uint32_t w = offsets[3]; w < shapes[3]; ++w)
                {
                    uint32_t i = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                    if (i >= n)
                    {
                        log_error("index %u at (%u, %u, %u, %u) out of range %u", i, b, c, h, w, n);
                        continue;
                    }

                    count++;
                    if (dtype == vkllm_dtype_float16)
                    {
                        float v0 = vkllm_fp16_to_fp32(lhs_fp16[i]);
                        float v1 = vkllm_fp16_to_fp32(rhs_fp16[i]);
                        err += (v0 - v1) * (v0 - v1);

                        if (fabsf(v0 - v1) > 1e-2 || isnan(err))
                        {
                            log_error("%s index %u at (%u, %u, %u, %u) err lhs %f rhs %f", name, i, b, c, h, w, v0, v1);
                            continue;
                        }
                    }
                    else
                    {
                        err += (lhs_fp32[i] - rhs_fp32[i]) * (lhs_fp32[i] - rhs_fp32[i]);
                        if (fabsf(lhs_fp32[i] - rhs_fp32[i]) > 1e-3 || isnan(err))
                        {
                            log_error("index %u at (%u, %u, %u, %u) err lhs %f rhs %f", i, b, c, h, w, lhs_fp32[i],
                                      rhs_fp32[i]);
                            continue;
                        }
                    }
                }
            }
        }
    }

    return count > 0 ? err / count : 0.0f;
}

static void update_rows_op_host(const void *input, void *output, const uint32_t in_shapes[4],
                                const uint32_t in_strides[4], const uint32_t out_shapes[4],
                                const uint32_t out_strides[4], uint32_t offset_rows, vkllm_dtype_t dtype)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);
    uint32_t dsize = info.bytes;

    const float *input_fp32 = (const float *)input;
    float *output_fp32 = (float *)output;

    const vkllm_fp16_pack *input_fp16 = (const vkllm_fp16_pack *)input;
    vkllm_fp16_pack *output_fp16 = (vkllm_fp16_pack *)output;

    uint32_t in_es[4] = {in_strides[0] / dsize, in_strides[1] / dsize, in_strides[2] / dsize, in_strides[3] / dsize};
    uint32_t out_es[4] = {out_strides[0] / dsize, out_strides[1] / dsize, out_strides[2] / dsize,
                          out_strides[3] / dsize};

    uint32_t B = in_shapes[0];
    uint32_t C = in_shapes[1];
    uint32_t H = in_shapes[2];
    uint32_t W = in_shapes[3];

    for (uint32_t b = 0; b < B; ++b)
    {
        for (uint32_t c = 0; c < C; ++c)
        {
            for (uint32_t h = 0; h < H; ++h)
            {
                for (uint32_t w = 0; w < W; ++w)
                {
                    uint32_t in_idx = b * in_es[0] + c * in_es[1] + h * in_es[2] + w * in_es[3];
                    uint32_t out_h = h + offset_rows;
                    uint32_t out_idx = b * out_es[0] + c * out_es[1] + out_h * out_es[2] + w * out_es[3];

                    if (dtype == vkllm_dtype_float32)
                    {
                        output_fp32[out_idx] = input_fp32[in_idx];
                    }
                    else if (dtype == vkllm_dtype_float16)
                    {
                        output_fp16[out_idx] = input_fp16[in_idx];
                    }
                }
            }
        }
    }
}

static struct
{
    uint32_t in_shapes[4];
    uint32_t out_shapes[4];
    uint32_t offset_rows;
    vkllm_dtype_t dtype;
} tests[] = {
    {{1, 1, 5, 32}, {1, 1, 10, 32}, 0, vkllm_dtype_float32}, {{1, 1, 5, 32}, {1, 1, 10, 32}, 5, vkllm_dtype_float32},
    {{2, 3, 4, 64}, {2, 3, 10, 64}, 2, vkllm_dtype_float32}, {{1, 2, 8, 128}, {1, 2, 16, 128}, 4, vkllm_dtype_float32},
    {{1, 1, 5, 32}, {1, 1, 10, 32}, 0, vkllm_dtype_float16}, {{1, 1, 5, 32}, {1, 1, 10, 32}, 5, vkllm_dtype_float16},
    {{2, 3, 4, 64}, {2, 3, 10, 64}, 2, vkllm_dtype_float16}, {{1, 2, 8, 128}, {1, 2, 16, 128}, 4, vkllm_dtype_float16},
};

START_TEST(test_op_update_rows)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *input;
    ck_assert_int_eq(vkllm_tensor_new(context, "input", tests[_i].in_shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, true, &input),
                     VKLLM_ERR_OK);

    struct vkllm_op_update_rows_params params = {.offset_rows = tests[_i].offset_rows};

    struct vkllm_tensor *srcs[] = {input};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", tests[_i].out_shapes, tests[_i].dtype, VKLLM_OP_UPDATE_ROWS,
                                      srcs, 1, (const uint8_t *)&params, sizeof(params), true, &output),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *input_host = NULL, *output_host = NULL;

    vkllm_array_u8_new(&input_host, input->bytes);
    vkllm_array_u8_new(&output_host, output->bytes);

    memset(output_host->data, 0, output_host->alloc_n);

    random_tensor(input_host->data, input->shapes, input->strides, input->dtype, -1.0, 1.0);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input, input_host->data, input_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_update_rows_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_update_rows_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_update_rows_post_run(context, commands, output), VKLLM_ERR_OK);

    update_rows_op_host(input_host->data, output_host->data, input->shapes, input->strides, output->shapes,
                        output->strides, tests[_i].offset_rows, input->dtype);

    const void *gpu_output = output->data.host;
    // print_first_n(context, commands, output, 0, 0, tests[_i].offset_rows, 32);

    float tolerance = (tests[_i].dtype == vkllm_dtype_float16) ? 1e-2 : 1e-4;
    char test_case_name[64];
    snprintf(test_case_name, sizeof(test_case_name), "test_op_update_rows_%d", _i);

    uint32_t compare_offsets[4] = {0, 0, tests[_i].offset_rows, 0};
    float error = compare_buf_offsets(output_host->data, gpu_output, output->shapes, output->strides, compare_offsets,
                                      output->bytes, output->dtype, test_case_name);

    if (error > tolerance)
    {
        log_error("Test case %d failed: error = %f, tolerance = %f", _i, error, tolerance);
        log_error("in_shapes: [%u, %u, %u, %u], out_shapes: [%u, %u, %u, %u], offset_rows: %u, dtype: %s",
                  tests[_i].in_shapes[0], tests[_i].in_shapes[1], tests[_i].in_shapes[2], tests[_i].in_shapes[3],
                  tests[_i].out_shapes[0], tests[_i].out_shapes[1], tests[_i].out_shapes[2], tests[_i].out_shapes[3],
                  tests[_i].offset_rows, vkllm_dtype_s(tests[_i].dtype));
    }

    ck_assert_float_le(error, tolerance);

    vkllm_tensor_free(context, input);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(input_host);
    vkllm_array_u8_free(output_host);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_update_rows_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase;
    suite = suite_create("vkllm_op_update_rows");
    tcase = tcase_create("vkllm_op_update_rows");

    tcase_add_loop_test(tcase, test_op_update_rows, 0, sizeof(tests) / sizeof(tests[0]));
    tcase_set_timeout(tcase, 60.0);
    suite_add_tcase(suite, tcase);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_update_rows_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
