#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_common.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_bin.h"
#include "src/vkllm_pipeline.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <stdio.h>
#include <string.h>

// ============================================================================
// CRITICAL FIX: Use a global context shared across all test cases
// This prevents repeated VkInstance creation/destruction which can cause
// GPU driver issues and device enumeration failures
// ============================================================================
static struct vkllm_context *g_context = NULL;

// Binary operations: C = A op B
// op: 0=add, 1=sub, 2=mul, 3=div
// A: [B, C, H, W], B: [B, C, H, W], C: [B, C, H, W]
static void bin_op_host(const void *input_a, const void *input_b, void *output, const uint32_t shapes[4],
                        const uint32_t strides_a[4], const uint32_t strides_b[4], const uint32_t strides_c[4],
                        vkllm_dtype_t dtype_a, vkllm_dtype_t dtype_b, int32_t op)
{
    struct vkllm_dtype_info info_a, info_b;
    vkllm_get_dtype_info(dtype_a, &info_a);
    vkllm_get_dtype_info(dtype_b, &info_b);

    // Convert byte strides to element strides
    uint32_t es_a[4] = {strides_a[0] / info_a.bytes, strides_a[1] / info_a.bytes, strides_a[2] / info_a.bytes,
                        strides_a[3] / info_a.bytes};
    uint32_t es_b[4] = {strides_b[0] / info_b.bytes, strides_b[1] / info_b.bytes, strides_b[2] / info_b.bytes,
                        strides_b[3] / info_b.bytes};
    uint32_t es_c[4] = {strides_c[0] / 4, strides_c[1] / 4, strides_c[2] / 4, strides_c[3] / 4};

    const vkllm_fp16_pack *a_fp16 = (const vkllm_fp16_pack *)input_a;
    const vkllm_fp16_pack *b_fp16 = (const vkllm_fp16_pack *)input_b;
    const float *a_fp32 = (const float *)input_a;
    const float *b_fp32 = (const float *)input_b;
    float *c_fp32 = (float *)output;

    for (uint32_t b = 0; b < shapes[0]; ++b)
    {
        for (uint32_t c = 0; c < shapes[1]; ++c)
        {
            for (uint32_t h = 0; h < shapes[2]; ++h)
            {
                for (uint32_t w = 0; w < shapes[3]; ++w)
                {
                    uint32_t idx_a = b * es_a[0] + c * es_a[1] + h * es_a[2] + w * es_a[3];
                    uint32_t idx_b = b * es_b[0] + c * es_b[1] + h * es_b[2] + w * es_b[3];
                    uint32_t idx_c = b * es_c[0] + c * es_c[1] + h * es_c[2] + w * es_c[3];

                    float val_a = 0.0f, val_b = 0.0f;

                    // Convert inputs to float32
                    if (dtype_a == vkllm_dtype_float16)
                    {
                        val_a = vkllm_fp16_to_fp32(a_fp16[idx_a]);
                    }
                    else
                    {
                        val_a = a_fp32[idx_a];
                    }

                    if (dtype_b == vkllm_dtype_float16)
                    {
                        val_b = vkllm_fp16_to_fp32(b_fp16[idx_b]);
                    }
                    else
                    {
                        val_b = b_fp32[idx_b];
                    }

                    // Perform operation
                    float result = 0.0f;
                    switch (op)
                    {
                    case 0: // add
                        result = val_a + val_b;
                        break;
                    case 1: // sub
                        result = val_a - val_b;
                        break;
                    case 2: // mul
                        result = val_a * val_b;
                        break;
                    case 3: // div
                        result = val_a / val_b;
                        break;
                    default:
                        result = 0.0f;
                        break;
                    }

                    c_fp32[idx_c] = result;
                }
            }
        }
    }
}

static const char *op_name(int32_t op)
{
    switch (op)
    {
    case 0:
        return "add";
    case 1:
        return "sub";
    case 2:
        return "mul";
    case 3:
        return "div";
    default:
        return "unknown";
    }
}

static struct
{
    uint32_t shapes[4];
    vkllm_dtype_t dtype_a;
    vkllm_dtype_t dtype_b;
    int32_t op; // 0: add, 1: sub, 2: mul, 3: div
} tests[] = {
    // Add operation tests
    {{1, 1, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 0},
    {{2, 4, 64, 128}, vkllm_dtype_float16, vkllm_dtype_float16, 0},
    {{1, 1, 128, 256}, vkllm_dtype_float32, vkllm_dtype_float32, 0},
    {{2, 4, 64, 128}, vkllm_dtype_float32, vkllm_dtype_float32, 0},
    {{4, 8, 32, 64}, vkllm_dtype_float32, vkllm_dtype_float32, 0},

    // Subtract operation tests
    {{1, 1, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 1},
    {{2, 4, 64, 128}, vkllm_dtype_float16, vkllm_dtype_float16, 1},
    {{1, 1, 128, 256}, vkllm_dtype_float32, vkllm_dtype_float32, 1},
    {{2, 4, 64, 128}, vkllm_dtype_float32, vkllm_dtype_float32, 1},
    {{4, 8, 32, 64}, vkllm_dtype_float32, vkllm_dtype_float32, 1},

    // Multiply operation tests
    {{1, 1, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 2},
    {{2, 4, 64, 128}, vkllm_dtype_float16, vkllm_dtype_float16, 2},
    {{1, 1, 128, 256}, vkllm_dtype_float32, vkllm_dtype_float32, 2},
    {{2, 4, 64, 128}, vkllm_dtype_float32, vkllm_dtype_float32, 2},
    {{4, 8, 32, 64}, vkllm_dtype_float32, vkllm_dtype_float32, 2},

    // Divide operation tests
    {{1, 1, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 3},
    {{2, 4, 64, 128}, vkllm_dtype_float16, vkllm_dtype_float16, 3},
    {{1, 1, 128, 256}, vkllm_dtype_float32, vkllm_dtype_float32, 3},
    {{2, 4, 64, 128}, vkllm_dtype_float32, vkllm_dtype_float32, 3},
    {{4, 8, 32, 64}, vkllm_dtype_float32, vkllm_dtype_float32, 3},

    // Large tensor tests for each operation
    {{8, 16, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 0},
    {{8, 16, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 1},
    {{8, 16, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 2},
    {{8, 16, 128, 256}, vkllm_dtype_float16, vkllm_dtype_float16, 3},
};

START_TEST(test_op_bin)
{
    // FIXED: Use global context instead of creating a new one each time
    struct vkllm_context *context = g_context;
    ck_assert_ptr_nonnull(context);

    struct vkllm_commands *commands;
    vkllm_err_t err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    uint32_t B = tests[_i].shapes[0];
    uint32_t C = tests[_i].shapes[1];
    uint32_t H = tests[_i].shapes[2];
    uint32_t W = tests[_i].shapes[3];
    int32_t op = tests[_i].op;

    // Create input tensor A
    struct vkllm_tensor *input_a;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_a", tests[_i].shapes, tests[_i].dtype_a, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &input_a),
                     VKLLM_ERR_OK);

    // Create input tensor B
    struct vkllm_tensor *input_b;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_b", tests[_i].shapes, tests[_i].dtype_b, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &input_b),
                     VKLLM_ERR_OK);

    // Create output tensor C with op parameter
    struct vkllm_tensor *srcs[] = {input_a, input_b};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", tests[_i].shapes, vkllm_dtype_float32, VKLLM_OP_BIN, srcs, 2,
                                      (const uint8_t *)&op, sizeof(op), true, &output),
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
    random_tensor(buf_a->data, input_a->shapes, input_a->strides, input_a->dtype, 0, 2.0);
    random_tensor(buf_b->data, input_b->shapes, input_b->strides, input_b->dtype, 1.0, 3.0);

    // Compute expected result on CPU
    bin_op_host(buf_a->data, buf_b->data, buf_c_expected->data, tests[_i].shapes, input_a->strides, input_b->strides,
                output->strides, tests[_i].dtype_a, tests[_i].dtype_b, op);

    // Execute on GPU
    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_a, buf_a->data, buf_a->alloc_n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_b, buf_b->data, buf_b->alloc_n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_bin_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_bin_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_bin_post_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);

    uint64_t time_cost = 0;
    ck_assert_int_eq(vkllm_pipeline_query_exec_time(context, output->pipeline, &time_cost), VKLLM_ERR_OK);

    log_info("test %d bin_op %s [%u,%u,%u,%u], dtype_a=%s, dtype_b=%s: avg time cost: %lu micro secs", _i,
             op_name(op), B, C, H, W, vkllm_dtype_s(tests[_i].dtype_a), vkllm_dtype_s(tests[_i].dtype_b),
             time_cost / 1000);

    // Compare results
    const void *gpu_output = output->data.host;

    // Use larger tolerance for float16 due to lower precision
    float tolerance = (tests[_i].dtype_a == vkllm_dtype_float16 || tests[_i].dtype_b == vkllm_dtype_float16) ? 1e-2
                                                                                                                : 1e-4;
    float error =
        compare_buf(buf_c_expected->data, gpu_output, output->shapes, output->strides, output->bytes, output->dtype);
    ck_assert_float_le(error, tolerance);

    // Clean up tensors (but NOT the context!)
    vkllm_tensor_free(context, input_a);
    vkllm_tensor_free(context, input_b);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(buf_a);
    vkllm_array_u8_free(buf_b);
    vkllm_array_u8_free(buf_c_expected);
    vkllm_commands_free(context, commands);

    // IMPORTANT: Do NOT call vkllm_context_free() here!
    // The context will be freed in the teardown function
}
END_TEST;

// Setup function called once before all tests
static void setup_global_context(void)
{
    vkllm_err_t err = vkllm_context_new(0, &g_context);
    if (err != VKLLM_ERR_OK)
    {
        fprintf(stderr, "Failed to create global context: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

// Teardown function called once after all tests
static void teardown_global_context(void)
{
    if (g_context != NULL)
    {
        vkllm_context_free(g_context);
        g_context = NULL;
    }
}

Suite *vkllm_op_bin_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase = NULL;
    suite = suite_create("vkllm_op_bin");
    tcase = tcase_create("vkllm_op_bin");

    // CRITICAL FIX: Use UNCHECKED fixtures to ensure setup/teardown runs ONLY ONCE
    // for the entire test suite, not once per test case
    // checked_fixture = runs before/after EACH test (wrong!)
    // unchecked_fixture = runs before/after ALL tests (correct!)
    tcase_add_unchecked_fixture(tcase, setup_global_context, teardown_global_context);

    tcase_add_loop_test(tcase, test_op_bin, 0, sizeof(tests) / sizeof(tests[0]));
    tcase_set_timeout(tcase, 120.0);
    suite_add_tcase(suite, tcase);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_bin_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
