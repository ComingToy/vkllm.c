#include "check.h"
#include "src/core/vkllm_array.h"
#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_op_rmsnorm.h"
#include "src/core/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <math.h>
#include <string.h>

// RMSNorm formula: y = x * w / sqrt(mean(x^2) + eps)
// where mean is computed over the last dimension (W)
static void rmsnorm_op_host(const void *input, const void *weight, void *output, const uint32_t shapes[4],
                            const uint32_t strides[4], const uint32_t weight_strides[4], vkllm_dtype_t input_dtype,
                            vkllm_dtype_t weight_dtype, vkllm_dtype_t output_dtype)
{
    struct vkllm_dtype_info input_info, weight_info, output_info;
    vkllm_get_dtype_info(input_dtype, &input_info);
    vkllm_get_dtype_info(weight_dtype, &weight_info);
    vkllm_get_dtype_info(output_dtype, &output_info);
    uint32_t input_dsize = input_info.bytes;
    uint32_t weight_dsize = weight_info.bytes;
    uint32_t output_dsize = output_info.bytes;

    const float *input_fp32 = (const float *)input;
    const float *weight_fp32 = (const float *)weight;
    float *output_fp32 = (float *)output;

    const vkllm_fp16_pack *input_fp16 = (const vkllm_fp16_pack *)input;
    const vkllm_fp16_pack *weight_fp16 = (const vkllm_fp16_pack *)weight;
    vkllm_fp16_pack *output_fp16 = (vkllm_fp16_pack *)output;

    float eps = 1e-6f;

    uint32_t in_es[4] = {strides[0] / input_dsize, strides[1] / input_dsize, strides[2] / input_dsize,
                         strides[3] / input_dsize};
    uint32_t w_es[4] = {weight_strides[0] / weight_dsize, weight_strides[1] / weight_dsize,
                        weight_strides[2] / weight_dsize, weight_strides[3] / weight_dsize};

    for (uint32_t b = 0; b < shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < shapes[2]; ++h)
            {
                // Compute RMS over the W dimension
                float sum_sq = 0.0f;
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t idx = b * in_es[0] + c * in_es[1] + h * in_es[2] + w * in_es[3];
                    float val = 0.0f;
                    if (input_dtype == vkllm_dtype_float32)
                    {
                        val = input_fp32[idx];
                    }
                    else if (input_dtype == vkllm_dtype_float16)
                    {
                        val = vkllm_fp16_to_fp32(input_fp16[idx]);
                    }
                    sum_sq += val * val;
                }

                float mean_sq = sum_sq / shapes[3];
                float rms = sqrtf(mean_sq + eps);

                // Normalize and scale
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t idx = b * in_es[0] + c * in_es[1] + h * in_es[2] + w * in_es[3];
                    uint32_t w_idx = w * w_es[3];

                    float val = 0.0f;
                    float weight_val = 0.0f;

                    if (input_dtype == vkllm_dtype_float32)
                    {
                        val = input_fp32[idx];
                    }
                    else if (input_dtype == vkllm_dtype_float16)
                    {
                        val = vkllm_fp16_to_fp32(input_fp16[idx]);
                    }

                    if (weight_dtype == vkllm_dtype_float32)
                    {
                        weight_val = weight_fp32[w_idx];
                    }
                    else if (weight_dtype == vkllm_dtype_float16)
                    {
                        weight_val = vkllm_fp16_to_fp32(weight_fp16[w_idx]);
                    }

                    if (output_dtype == vkllm_dtype_float32)
                    {
                        output_fp32[idx] = (val / rms) * weight_val;
                    }
                    else if (output_dtype == vkllm_dtype_float16)
                    {
                        output_fp16[idx] = vkllm_fp32_to_fp16((val / rms) * weight_val);
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
    // Float16 tests
    {{1, 1, 10, 128}, vkllm_dtype_float16},
    {{2, 1, 5, 256}, vkllm_dtype_float16},
    {{1, 2, 8, 512}, vkllm_dtype_float16},
    {{3, 4, 6, 64}, vkllm_dtype_float16},
};

START_TEST(test_op_rmsnorm)
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

    // Create weight tensor (shape: 1, 1, 1, W)
    // Weight tensor must be float32 for f16f32f16 shader
    uint32_t weight_shapes[4] = {1, 1, 1, tests[_i].shapes[3]};
    struct vkllm_tensor *weight;
    ck_assert_int_eq(vkllm_tensor_new(context, "weight", weight_shapes, vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &weight),
                     VKLLM_ERR_OK);

    // Create output tensor
    struct vkllm_tensor *srcs[] = {input, weight};
    struct vkllm_op_rmsnorm_params rmsnorm_params = {2.0f, 1e-6f};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_RMSNORM, srcs, 2,
                                      &rmsnorm_params, sizeof(rmsnorm_params), true, &output),
                     VKLLM_ERR_OK);

    // Allocate host buffers
    struct vkllm_array_u8 *input_host = NULL, *weight_host = NULL, *output_host = NULL;

    vkllm_array_u8_new(&input_host, input->bytes);
    vkllm_array_u8_new(&weight_host, weight->bytes);
    vkllm_array_u8_new(&output_host, output->bytes);

    memset(output_host->data, 0, output_host->alloc_n);

    // Generate random data
    random_tensor(input_host->data, input->shapes, input->strides, input->dtype, -1.0, 1.0);
    random_tensor(weight_host->data, weight->shapes, weight->strides, weight->dtype, -1.0, 1.0);

    // Upload data and execute
    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input, input_host->data, input_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, weight, weight_host->data, weight_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_rmsnorm_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_rmsnorm_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_rmsnorm_post_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);

    // Compute expected result on CPU
    rmsnorm_op_host(input_host->data, weight_host->data, output_host->data, input->shapes, input->strides,
                    weight->strides, input->dtype, weight->dtype, output->dtype);

    const void *gpu_output = output->data.host;

    // Compare results (allow slightly larger tolerance for fp16)

    char test_case_name[64];
    snprintf(test_case_name, sizeof(test_case_name), "test_op_rmsnorm_%d", _i);
    float tolerance = (tests[_i].dtype == vkllm_dtype_float16) ? 1e-2 : 1e-4;
    ck_assert_float_le(compare_buf(output_host->data, gpu_output, output->shapes, output->strides, output->bytes,
                                   output->dtype, test_case_name),
                       tolerance);

    // Clean up
    vkllm_tensor_free(context, input);
    vkllm_tensor_free(context, weight);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(input_host);
    vkllm_array_u8_free(weight_host);
    vkllm_array_u8_free(output_host);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_rmsnorm_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase_f16 = NULL;
    suite = suite_create("vkllm_op_rmsnorm");
    tcase_f16 = tcase_create("vkllm_op_rmsnorm_f16");

    tcase_add_loop_test(tcase_f16, test_op_rmsnorm, 0, 4);
    tcase_set_timeout(tcase_f16, 60.0);
    suite_add_tcase(suite, tcase_f16);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_rmsnorm_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
