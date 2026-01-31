#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_softmax.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <math.h>
#include <string.h>

// Softmax formula: y[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
// Computed over the last dimension (W) for numerical stability
static void softmax_op_host(const void *input, void *output, const uint32_t shapes[4], const uint32_t strides[4],
                            vkllm_dtype_t dtype, int32_t seq_mask, uint32_t offsets)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);
    uint32_t dsize = info.bytes;

    const float *input_fp32 = (const float *)input;
    float *output_fp32 = (float *)output;

    const vkllm_fp16_pack *input_fp16 = (const vkllm_fp16_pack *)input;
    vkllm_fp16_pack *output_fp16 = (vkllm_fp16_pack *)output;

    uint32_t es[4] = {strides[0] / dsize, strides[1] / dsize, strides[2] / dsize, strides[3] / dsize};

    for (uint32_t b = 0; b < shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < shapes[2]; ++h)
            {
                // Find maximum value in the row for numerical stability
                float max_val = -INFINITY;
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    // Skip masked positions if seq_mask is enabled
                    if (seq_mask > 0 && w > h + offsets)
                    {
                        continue;
                    }

                    uint32_t idx = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                    float val = 0.0f;
                    if (dtype == vkllm_dtype_float32)
                    {
                        val = input_fp32[idx];
                    }
                    else if (dtype == vkllm_dtype_float16)
                    {
                        val = vkllm_fp16_to_fp32(input_fp16[idx]);
                    }
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }

                // Compute sum of exp(x - max)
                float sum_exp = 0.0f;
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    // Skip masked positions
                    if (seq_mask > 0 && w > h + offsets)
                    {
                        continue;
                    }

                    uint32_t idx = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                    float val = 0.0f;
                    if (dtype == vkllm_dtype_float32)
                    {
                        val = input_fp32[idx];
                    }
                    else if (dtype == vkllm_dtype_float16)
                    {
                        val = vkllm_fp16_to_fp32(input_fp16[idx]);
                    }
                    sum_exp += expf(val - max_val);
                }

                // Normalize
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t idx = b * es[0] + c * es[1] + h * es[2] + w * es[3];

                    if (seq_mask > 0 && w > h + offsets)
                    {
                        // Masked positions get 0
                        if (dtype == vkllm_dtype_float32)
                        {
                            output_fp32[idx] = 0.0f;
                        }
                        else if (dtype == vkllm_dtype_float16)
                        {
                            output_fp16[idx] = vkllm_fp32_to_fp16(0.0f);
                        }
                    }
                    else
                    {
                        float val = 0.0f;
                        if (dtype == vkllm_dtype_float32)
                        {
                            val = input_fp32[idx];
                            output_fp32[idx] = expf(val - max_val) / sum_exp;
                        }
                        else if (dtype == vkllm_dtype_float16)
                        {
                            val = vkllm_fp16_to_fp32(input_fp16[idx]);
                            output_fp16[idx] = vkllm_fp32_to_fp16(expf(val - max_val) / sum_exp);
                        }
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
    int32_t seq_mask;
    uint32_t offsets;
} tests[] = {
    // Float32 tests without masking
    {{1, 1, 10, 128}, vkllm_dtype_float32, 0, 0},
    {{2, 1, 5, 256}, vkllm_dtype_float32, 0, 0},
    {{1, 2, 8, 512}, vkllm_dtype_float32, 0, 0},
    {{3, 4, 6, 64}, vkllm_dtype_float32, 0, 0},

    // Float32 tests with masking
    {{1, 1, 10, 128}, vkllm_dtype_float32, 1, 0},
    {{2, 1, 5, 256}, vkllm_dtype_float32, 1, 0},

    // // Float16 tests without masking
    {{1, 1, 10, 128}, vkllm_dtype_float16, 0, 0},
    {{2, 1, 5, 256}, vkllm_dtype_float16, 0, 0},
    {{1, 2, 8, 512}, vkllm_dtype_float16, 0, 0},

    // // Float16 tests with masking
    {{1, 1, 10, 128}, vkllm_dtype_float16, 1, 0},
};

START_TEST(test_op_softmax)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Create input tensor
    struct vkllm_tensor *input;
    ck_assert_int_eq(vkllm_tensor_new(context, "input", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL,
                                      0, false, &input),
                     VKLLM_ERR_OK);

    // Create output tensor with params (seq_mask and offsets)
    struct vkllm_tensor *srcs[] = {input};

    // Pack params: [seq_mask (int32), offsets (uint32)]
    struct vkllm_op_softmax_params params = {.seq_mask = tests[_i].seq_mask, .offsets = tests[_i].offsets};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_SOFTMAX, srcs, 1,
                                      &params, sizeof(params), true, &output),
                     VKLLM_ERR_OK);

    // Allocate host buffers
    struct vkllm_array_u8 *input_host = NULL, *output_host = NULL;

    vkllm_array_u8_new(&input_host, input->bytes);
    vkllm_array_u8_new(&output_host, output->bytes);

    memset(output_host->data, 0, output_host->alloc_n);

    // Generate random data (use smaller range for better numerical stability)
    random_tensor(input_host->data, input->shapes, input->strides, input->dtype, -1.0, 1.0);

    // Upload data and execute
    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input, input_host->data, input_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_softmax_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_softmax_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_softmax_post_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);

    // Compute expected result on CPU
    softmax_op_host(input_host->data, output_host->data, input->shapes, input->strides, input->dtype,
                    tests[_i].seq_mask, tests[_i].offsets);

    const void *gpu_output = output->data.host;

#ifdef __VKLLM_DEBUG__
    print_n("gpu output", gpu_output, 32);
    print_n("host output", (const float *)output_host->data, 32);
#endif

    // Compare results (allow slightly larger tolerance for fp16 and for exp/division operations)
    float tolerance = (tests[_i].dtype == vkllm_dtype_float16) ? 1e-2 : 1e-4;
    float error =
        compare_buf(output_host->data, gpu_output, output->shapes, output->strides, output->bytes, output->dtype);

    ck_assert_float_le(error, tolerance);

    // Verify that each row sums to approximately 1.0 (for non-masked positions)
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(tests[_i].dtype, &info);
    uint32_t es[4] = {output->strides[0] / info.bytes, output->strides[1] / info.bytes, output->strides[2] / info.bytes,
                      output->strides[3] / info.bytes};

    const float *out_fp32 = (const float *)gpu_output;
    const vkllm_fp16_pack *out_fp16 = (const vkllm_fp16_pack *)gpu_output;

    for (uint32_t b = 0; b < tests[_i].shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < tests[_i].shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < tests[_i].shapes[2]; ++h)
            {
                float row_sum = 0.0f;
                for (uint32_t w = 0; w < tests[_i].shapes[3]; ++w)
                {
                    if (tests[_i].seq_mask > 0 && w > h + tests[_i].offsets)
                    {
                        continue;
                    }

                    uint32_t idx = b * es[0] + c * es[1] + h * es[2] + w * es[3];
                    if (tests[_i].dtype == vkllm_dtype_float32)
                    {
                        row_sum += out_fp32[idx];
                    }
                    else
                    {
                        row_sum += vkllm_fp16_to_fp32(out_fp16[idx]);
                    }
                }

#ifdef __VKLLM_DEBUG__
                fprintf(stderr, "row (%u, %u, %u) sum = %f\n", b, c, h, row_sum);
#endif
                // Each row should sum to approximately 1.0
                ck_assert_float_le(fabsf(row_sum - 1.0f), 5e-2);
            }
        }
    }

    // Clean up
    vkllm_tensor_free(context, input);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(input_host);
    vkllm_array_u8_free(output_host);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_softmax_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase = NULL;
    suite = suite_create("vkllm_op_softmax");

    tcase = tcase_create("vkllm_op_softmax");
    tcase_add_loop_test(tcase, test_op_softmax, 0, sizeof(tests) / sizeof(tests[0]));
    tcase_set_timeout(tcase, 60.0);

    suite_add_tcase(suite, tcase);

    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_softmax_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
