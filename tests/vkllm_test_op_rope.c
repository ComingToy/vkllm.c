#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_rope.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <math.h>
#include <string.h>

// RoPE formula: 
// For each pair (x[2i], x[2i+1]) at position pos:
// theta = (pos + offset) / (base ^ (2i / dim))
// x'[2i]   = cos(theta) * x[2i]   - sin(theta) * x[2i+1]
// x'[2i+1] = sin(theta) * x[2i]   + cos(theta) * x[2i+1]
static void rope_op_host(const void *input, void *output, const uint32_t shapes[4], const uint32_t strides[4],
                         uint32_t offset, float base, vkllm_dtype_t dtype)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);
    uint32_t dsize = info.bytes;

    const float *input_fp32 = (const float *)input;
    float *output_fp32 = (float *)output;

    const vkllm_fp16_pack *input_fp16 = (const vkllm_fp16_pack *)input;
    vkllm_fp16_pack *output_fp16 = (vkllm_fp16_pack *)output;

    uint32_t es[4] = {strides[0] / dsize, strides[1] / dsize, strides[2] / dsize, strides[3] / dsize};

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
                // Process pairs along the W dimension
                for (uint32_t w = 0; w < W / 2; ++w)
                {
                    // Calculate indices for the pair
                    uint32_t idx0 = b * es[0] + c * es[1] + h * es[2] + (2 * w) * es[3];
                    uint32_t idx1 = idx0 + es[3];

                    // Get input values
                    float v0 = 0.0f, v1 = 0.0f;
                    if (dtype == vkllm_dtype_float32)
                    {
                        v0 = input_fp32[idx0];
                        v1 = input_fp32[idx1];
                    }
                    else if (dtype == vkllm_dtype_float16)
                    {
                        v0 = vkllm_fp16_to_fp32(input_fp16[idx0]);
                        v1 = vkllm_fp16_to_fp32(input_fp16[idx1]);
                    }

                    // Calculate rotation angle
                    // theta = (pos + offset) / (base ^ (2*w / W))
                    float freq = (float)(offset + h) / powf(base, (float)(2 * w) / (float)W);
                    float cos_theta = cosf(freq);
                    float sin_theta = sinf(freq);

                    // Apply rotation
                    float out0 = cos_theta * v0 - sin_theta * v1;
                    float out1 = sin_theta * v0 + cos_theta * v1;

                    // Write output
                    if (dtype == vkllm_dtype_float32)
                    {
                        output_fp32[idx0] = out0;
                        output_fp32[idx1] = out1;
                    }
                    else if (dtype == vkllm_dtype_float16)
                    {
                        output_fp16[idx0] = vkllm_fp32_to_fp16(out0);
                        output_fp16[idx1] = vkllm_fp32_to_fp16(out1);
                    }
                }
            }
        }
    }
}

static struct
{
    uint32_t shapes[4];
    uint32_t offset;
    float base;
    vkllm_dtype_t dtype;
} tests[] = {
    // Float32 tests
    {{1, 1, 10, 128}, 0, 10000.0f, vkllm_dtype_float32},
    {{2, 1, 5, 64}, 0, 10000.0f, vkllm_dtype_float32},
    {{1, 2, 8, 256}, 5, 10000.0f, vkllm_dtype_float32},
    {{3, 4, 6, 32}, 10, 10000.0f, vkllm_dtype_float32},
    // Float16 tests
    {{8, 32, 32, 128}, 0, 10000.0f, vkllm_dtype_float16},
    {{2, 1, 5, 64}, 0, 10000.0f, vkllm_dtype_float16},
    {{1, 2, 8, 256}, 5, 10000.0f, vkllm_dtype_float16},
    {{3, 4, 6, 32}, 10, 10000.0f, vkllm_dtype_float16},
};

START_TEST(test_op_rope)
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

    // Create RoPE parameters
    struct vkllm_rope_params params = {
        .offset = tests[_i].offset,
        .base = tests[_i].base,
    };

    // Create output tensor with RoPE operation
    struct vkllm_tensor *srcs[] = {input};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_ROPE, srcs, 1,
                                      (const uint8_t *)&params, sizeof(params), true, &output),
                     VKLLM_ERR_OK);

    // Allocate host buffers
    struct vkllm_array_u8 *input_host = NULL, *output_host = NULL;

    vkllm_array_u8_new(&input_host, input->bytes);
    vkllm_array_u8_new(&output_host, output->bytes);

    memset(output_host->data, 0, output_host->alloc_n);

    // Generate random data
    random_tensor(input_host->data, input->shapes, input->strides, input->dtype);
	// print_n_f16("input host: ", input_host->data, 64);

    // Upload data and execute
    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input, input_host->data, input_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_rope_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_rope_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_rope_post_run(context, commands, output), VKLLM_ERR_OK);

    // Compute expected result on CPU
    rope_op_host(input_host->data, output_host->data, input->shapes, input->strides, tests[_i].offset, tests[_i].base,
                 input->dtype);

    const void *gpu_output = output->data.host;
    // print_n("gpu output: ", gpu_output, 32);
    // print_n("host output: ", (float*)output_host->data, 32);

    // Compare results (allow slightly larger tolerance for fp16)
    float tolerance = (tests[_i].dtype == vkllm_dtype_float16) ? 1e-2 : 1e-4;
    float error =
        compare_buf(output_host->data, gpu_output, output->shapes, output->strides, output->bytes, output->dtype);
    
    // Debug output if test fails
    if (error > tolerance)
    {
        log_error("Test case %d failed: error = %f, tolerance = %f", _i, error, tolerance);
        log_error("Shapes: [%u, %u, %u, %u], offset: %u, base: %f, dtype: %s", 
                  tests[_i].shapes[0], tests[_i].shapes[1], tests[_i].shapes[2], tests[_i].shapes[3],
                  tests[_i].offset, tests[_i].base, vkllm_dtype_s(tests[_i].dtype));
    }

    ck_assert_float_le(error, tolerance);

    // Clean up
    vkllm_tensor_free(context, input);
    vkllm_tensor_free(context, output);
    vkllm_array_u8_free(input_host);
    vkllm_array_u8_free(output_host);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_rope_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase;
    suite = suite_create("vkllm_op_rope");
    tcase = tcase_create("vkllm_op_rope_f32");

    tcase_add_loop_test(tcase, test_op_rope, 0, sizeof(tests)/sizeof(tests[0]));
    tcase_set_timeout(tcase, 60.0);
    suite_add_tcase(suite, tcase);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_rope_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
