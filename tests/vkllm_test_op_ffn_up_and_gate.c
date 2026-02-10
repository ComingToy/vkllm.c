#include "check.h"
#include "src/core/vkllm_array.h"
#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_op_ffn_up_and_gate.h"
#include "src/core/vkllm_pipeline.h"
#include "src/core/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static struct vkllm_context *g_context = NULL;

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

START_TEST(test_ffn_up_and_gate_f32)
{
    struct vkllm_context *context = g_context;

    struct vkllm_commands *commands;
    ck_assert_int_eq(vkllm_commands_new(context, &commands), VKLLM_ERR_OK);

    uint32_t B = 1;
    uint32_t C = 1;
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 256;

    uint32_t shapes_x[4] = {B, C, M, K};
    uint32_t shapes_w_up[4] = {1, 1, N, K};
    uint32_t shapes_w_gate[4] = {1, 1, N, K};
    uint32_t shapes_out[4] = {B, C, M, N};

    struct vkllm_tensor *input_x;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_x", shapes_x, vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL,
                                      0, false, &input_x),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *input_w_up;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_w_up", shapes_w_up, vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &input_w_up),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *input_w_gate;
    ck_assert_int_eq(vkllm_tensor_new(context, "input_w_gate", shapes_w_gate, vkllm_dtype_float32, VKLLM_OP_NONE, NULL,
                                      0, NULL, 0, false, &input_w_gate),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *srcs[] = {input_x, input_w_up, input_w_gate};
    struct vkllm_tensor *output;
    ck_assert_int_eq(vkllm_tensor_new(context, "output", shapes_out, vkllm_dtype_float32, VKLLM_OP_FFN_UP_AND_GATE,
                                      srcs, 3, NULL, 0, true, &output),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *buf_x = NULL, *buf_w_up = NULL, *buf_w_gate = NULL, *buf_out_expected = NULL;

    vkllm_array_u8_new(&buf_x, input_x->bytes);
    vkllm_array_u8_new(&buf_w_up, input_w_up->bytes);
    vkllm_array_u8_new(&buf_w_gate, input_w_gate->bytes);
    vkllm_array_u8_new(&buf_out_expected, output->bytes);

    memset(buf_x->data, 0, buf_x->alloc_n);
    memset(buf_w_up->data, 0, buf_w_up->alloc_n);
    memset(buf_w_gate->data, 0, buf_w_gate->alloc_n);
    memset(buf_out_expected->data, 0, buf_out_expected->alloc_n);

    random_tensor(buf_x->data, input_x->shapes, input_x->strides, input_x->dtype, -1.0, 1.0);
    random_tensor(buf_w_up->data, input_w_up->shapes, input_w_up->strides, input_w_up->dtype, -1.0, 1.0);
    random_tensor(buf_w_gate->data, input_w_gate->shapes, input_w_gate->strides, input_w_gate->dtype, -1.0, 1.0);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_x, buf_x->data, buf_x->alloc_n), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_w_up, buf_w_up->data, buf_w_up->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, input_w_gate, buf_w_gate->data, buf_w_gate->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_ffn_up_and_gate_init(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_ffn_up_and_gate_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_ffn_up_and_gate_post_run(context, commands, output), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);

    vkllm_tensor_invalid_cache(context, output);
    const void *gpu_output = output->data.host;
    __UNUSED(gpu_output);

    vkllm_array_u8_free(buf_x);
    vkllm_array_u8_free(buf_w_up);
    vkllm_array_u8_free(buf_w_gate);
    vkllm_array_u8_free(buf_out_expected);
    vkllm_tensor_free(context, input_x);
    vkllm_tensor_free(context, input_w_up);
    vkllm_tensor_free(context, input_w_gate);
    vkllm_tensor_free(context, output);
    vkllm_commands_free(context, commands);
}
END_TEST

Suite *vkllm_test_op_ffn_up_and_gate_suite(void)
{
    Suite *s = suite_create("vkllm_test_op_ffn_up_and_gate");
    TCase *tc = tcase_create("ffn_up_and_gate");

    tcase_add_unchecked_fixture(tc, setup_global_context, teardown_global_context);
    tcase_add_test(tc, test_ffn_up_and_gate_f32);
    tcase_set_timeout(tc, 120.0);

    suite_add_tcase(s, tc);
    return s;
}

int main(void)
{
    int number_failed = 0;

    Suite *s = vkllm_test_op_ffn_up_and_gate_suite();
    SRunner *sr = srunner_create(s);
    srunner_set_fork_status(sr, CK_NOFORK);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? 0 : 1;
}
