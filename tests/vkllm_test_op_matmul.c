#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_matmul.h"
#include "src/vkllm_pipeline.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <stdio.h>
#include <string.h>

// Matrix multiplication: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// Using 4D tensor format: [1, 1, M, K/N]
static void matmul_op_host(const void *input_a, const void *input_b, void *output, uint32_t M, uint32_t K, uint32_t N,
                           vkllm_dtype_t dtype)
{
    const float *a_fp32 = (const float *)input_a;
    const float *b_fp32 = (const float *)input_b;
    float *c_fp32 = (float *)output;

    // C[i][j] = sum(A[i][k] * B[k][j]) for k in [0, K)
    for (uint32_t i = 0; i < M; ++i)
    {
        for (uint32_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
            {
                sum += a_fp32[i * K + k] * b_fp32[j * K + k];
            }
            c_fp32[i * N + j] = sum;
        }
    }
}

static struct
{
    uint32_t M;
    uint32_t K;
    uint32_t N;
    vkllm_dtype_t dtype;
} tests[] = {
    // Larger matrices
    {2048, 1024, 10240, vkllm_dtype_float32},
    {512, 1024, 2048, vkllm_dtype_float32},
    // {512, 1024, 2048, vkllm_dtype_float32},
};

START_TEST(test_op_matmul)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    uint32_t M = tests[_i].M;
    uint32_t K = tests[_i].K;
    uint32_t N = tests[_i].N;

    // Create input tensor A: shape [1, 1, M, K]
    uint32_t shapes_a[4] = {1, 1, M, K};
    struct vkllm_tensor *input_a;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_a", shapes_a, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL, 0,
                                      false, &input_a),
                     VKLLM_ERR_OK);

    // Create input tensor B: shape [1, 1, K, N]
    uint32_t shapes_b[4] = {1, 1, N, K};
    struct vkllm_tensor *input_b;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_b", shapes_b, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0, NULL, 0,
                                      false, &input_b),
                     VKLLM_ERR_OK);

    // Create output tensor C: shape [1, 1, M, N]
    uint32_t shapes_c[4] = {1, 1, M, N};
    struct vkllm_tensor *srcs[] = {input_a, input_b};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", shapes_c, vkllm_dtype_float32, VKLLM_OP_MATMUL, srcs, 2, NULL,
                                      0, true, &output),
                     VKLLM_ERR_OK);

    // Allocate host buffers
    struct vkllm_array_u8 *buf_a = NULL, *buf_b = NULL, *buf_c_expected = NULL;

    vkllm_array_u8_new(&buf_a, input_a->bytes);
    vkllm_array_u8_new(&buf_b, input_b->bytes);
    vkllm_array_u8_new(&buf_c_expected, output->bytes);

    memset(buf_a->data, 0, buf_a->alloc_n);
    memset(buf_b->data, 0, buf_b->alloc_n);
    memset(buf_c_expected->data, 0, buf_c_expected->alloc_n);

    // Generate random input data
    random_tensor(buf_a->data, input_a->shapes, input_a->strides, input_a->dtype);
    random_tensor(buf_b->data, input_b->shapes, input_b->strides, input_b->dtype);

    // Compute expected result on CPU
    matmul_op_host(buf_a->data, buf_b->data, buf_c_expected->data, M, K, N, tests[_i].dtype);

    // Execute on GPU

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_a, buf_a->data, buf_a->alloc_n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_b, buf_b->data, buf_b->alloc_n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);

    uint64_t total_time_cost = 0;
    for (uint32_t i = 0; i < 50; ++i)
    {
        ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
        ck_assert_int_eq(vkllm_op_matmul(context, commands, output), VKLLM_ERR_OK);
        ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
        ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
        ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
        ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);

        uint64_t time_cost = 0;
        ck_assert_int_eq(vkllm_pipeline_query_exec_time(context, output->pipeline, &time_cost), VKLLM_ERR_OK);
        total_time_cost += time_cost;
    }

    log_info("matmul: avg time cost: %lu micro secs", total_time_cost / 50 / 1000);
    // Compare results
    const void *gpu_output = output->data.host;
    float tolerance = 1e-3; // Slightly larger tolerance for matmul due to accumulation errors
    ck_assert_float_le(
        compare_buf(buf_c_expected->data, gpu_output, output->shapes, output->strides, output->bytes, output->dtype),
        tolerance);

    // Clean up
    vkllm_tensor_free(context, input_a);
    vkllm_tensor_free(context, input_b);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(buf_a);
    vkllm_array_u8_free(buf_b);
    vkllm_array_u8_free(buf_c_expected);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_matmul_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase_f32 = NULL;
    suite = suite_create("vkllm_op_matmul");
    tcase_f32 = tcase_create("vkllm_op_matmul_f32");

    tcase_add_loop_test(tcase_f32, test_op_matmul, 0, 2);
    tcase_set_timeout(tcase_f32, 120.0);
    suite_add_tcase(suite, tcase_f32);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_matmul_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
