#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_add.h"
#include "src/vkllm_op_embedding.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <string.h>
#

static struct
{
    uint32_t shapes0[4];
    uint32_t shapes1[4];
    vkllm_dtype_t dtype;
} tests_f32[] = {
    {.shapes0 = {1, 1, 32, 32}, .shapes1 = {1, 1, 128, 64}, .dtype = vkllm_dtype_float32},
    {.shapes0 = {1, 1, 32, 99}, .shapes1 = {1, 1, 128, 64}, .dtype = vkllm_dtype_float32},
};

static struct
{
    uint32_t shapes0[4];
    uint32_t shapes1[4];
    vkllm_dtype_t dtype;
} tests_f16[] = {
    {.shapes0 = {1, 1, 32, 32}, .shapes1 = {1, 1, 128, 64}, .dtype = vkllm_dtype_float16},
    {.shapes0 = {1, 1, 32, 99}, .shapes1 = {1, 1, 128, 64}, .dtype = vkllm_dtype_float16},
};

static void embedding_op_host(const uint32_t *indices, const void *params, const void *out, const uint32_t shapes[4],
                              const uint32_t shapes1[4], const uint32_t shapes2[4], const uint32_t strides[4],
                              const uint32_t strides1[4], const uint32_t strides2[4], vkllm_dtype_t dtype)
{
    struct vkllm_dtype_info info;
    vkllm_get_dtype_info(dtype, &info);
    uint32_t dsize = info.bytes;

    float *out_fp32 = (float *)out;
    float *params_fp32 = (float *)params;
    vkllm_fp16_pack *out_fp16 = (vkllm_fp16_pack *)out;
    vkllm_fp16_pack *params_fp16 = (vkllm_fp16_pack *)params;
    uint32_t n_params = strides1[0] * shapes1[0] / dsize;
    uint32_t n_out = strides2[0] * shapes2[0] / dsize;
    do
    {
        for (uint32_t _b = 0; _b < shapes[0]; ++_b)
        {
            for (uint32_t _c = 0; _c < shapes[1]; ++_c)
            {
                for (uint32_t _h = 0; _h < shapes[2]; ++_h)
                {
                    for (uint32_t _w = 0; _w < shapes[3]; ++_w)
                    {
                        {
                            uint32_t i = get_indice(_b, _c, _h, _w, strides, sizeof(uint32_t));
                            uint32_t tok = indices[i];
                            if (tok >= shapes1[2])
                            {
                                tok = 0;
                            }
                            uint32_t params_off = get_indice(0, 0, tok, 0, strides1, dsize);
                            uint32_t out_off = get_indice(_c, _h, _w, 0, strides2, dsize);
                            for (uint32_t k = 0; k < shapes1[3]; ++k)
                            {
                                if (out_off + k >= n_out)
                                {
                                    log_error("tok %u indices %u + %u = %u at (%u %u %u %u) out of output0 %u", tok, k,
                                              out_off, k + out_off, _b, _c, _h, _w, n_out);
                                }

                                if (params_off + k >= n_params)
                                {
                                    log_error("tok %u indices %u + %u = %u at (%u %u %u %u) out of params %u", tok, k,
                                              params_off, k + params_off, _b, _c, _h, _w, n_params);
                                }

                                if (dtype == vkllm_dtype_float32)
                                {
                                    out_fp32[out_off + k] = params_fp32[params_off + k];
                                }
                                else if (dtype == vkllm_dtype_float16)
                                {
                                    out_fp16[out_off + k] = params_fp16[params_off + k];
                                }
                            }
                        };
                    }
                }
            }
        }
    } while (0);
}

START_TEST(test_embedding_op_f32)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *indices = NULL, *params = NULL;
    ck_assert_int_eq(vkllm_tensor_new(context, "indices", tests_f32[_i].shapes0, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &indices),
                     VKLLM_ERR_OK);

    ck_assert_int_eq(vkllm_tensor_new(context, "params", tests_f32[_i].shapes1, tests_f32[_i].dtype, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &params),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *out0 = NULL;
    uint32_t shapes_out[] = {tests_f32[_i].shapes0[1], tests_f32[_i].shapes0[2], tests_f32[_i].shapes0[3], tests_f32[_i].shapes1[3]};
    struct vkllm_tensor *srcs[] = {indices, params};

    uint32_t UNK_TOK = 0;
    ck_assert_int_eq(vkllm_tensor_new(context, "out0", shapes_out, tests_f32[_i].dtype, VKLLM_OP_EMBEDDING, srcs, 2,
                                      (const uint8_t *)&UNK_TOK, sizeof(UNK_TOK), true, &out0),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *indices_host = NULL, *params_host = NULL, *out0_host = NULL;

    vkllm_array_u8_new(&indices_host, indices->bytes);
    vkllm_array_u8_new(&params_host, params->bytes);
    vkllm_array_u8_new(&out0_host, out0->bytes);

    memset(out0_host->data, 0, out0_host->alloc_n);

    random_tensor(indices_host->data, indices->shapes, indices->strides, indices->dtype);
    random_tensor(params_host->data, params->shapes, params->strides, params->dtype);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, indices, indices_host->data, indices_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, params, params_host->data, params_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_embedding(context, commands, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, out0), VKLLM_ERR_OK);

    embedding_op_host((const uint32_t *)indices_host->data, params_host->data, out0_host->data, indices->shapes,
                      params->shapes, out0->shapes, indices->strides, params->strides, out0->strides, params->dtype);

    const void *p = out0->data.host;

    // print_n("out0 host", (const float *)out0_host->data, 512);
    // print_n("out0", p, 512);

    ck_assert_float_le(compare_buf(out0_host->data, p, out0->shapes, out0->strides, out0->bytes, out0->dtype), 1e-5);

    vkllm_tensor_free(context, out0);
    vkllm_tensor_free(context, indices);
    vkllm_tensor_free(context, params);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
    vkllm_array_u8_free(indices_host);
    vkllm_array_u8_free(params_host);
}
END_TEST;

START_TEST(test_embedding_op_f16)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *indices = NULL, *params = NULL;
    ck_assert_int_eq(vkllm_tensor_new(context, "indices", tests_f16[_i].shapes0, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &indices),
                     VKLLM_ERR_OK);

    ck_assert_int_eq(vkllm_tensor_new(context, "params", tests_f16[_i].shapes1, tests_f16[_i].dtype, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &params),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *out0 = NULL;
    uint32_t shapes_out[] = {tests_f16[_i].shapes0[1], tests_f16[_i].shapes0[2], tests_f16[_i].shapes0[3], tests_f16[_i].shapes1[3]};
    struct vkllm_tensor *srcs[] = {indices, params};

    uint32_t UNK_TOK = 0;
    ck_assert_int_eq(vkllm_tensor_new(context, "out0", shapes_out, tests_f16[_i].dtype, VKLLM_OP_EMBEDDING, srcs, 2,
                                      (const uint8_t *)&UNK_TOK, sizeof(UNK_TOK), true, &out0),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *indices_host = NULL, *params_host = NULL, *out0_host = NULL;

    vkllm_array_u8_new(&indices_host, indices->bytes);
    vkllm_array_u8_new(&params_host, params->bytes);
    vkllm_array_u8_new(&out0_host, out0->bytes);

    memset(out0_host->data, 0, out0_host->alloc_n);

    random_tensor(indices_host->data, indices->shapes, indices->strides, indices->dtype);
    random_tensor(params_host->data, params->shapes, params->strides, params->dtype);

    ck_assert_int_eq(vkllm_commands_begin(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, indices, indices_host->data, indices_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_upload(context, commands, params, params_host->data, params_host->alloc_n),
                     VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_op_embedding(context, commands, out0), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_end(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_submit(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_commands_wait_exec(context, commands), VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_tensor_flush_cache(context, out0), VKLLM_ERR_OK);

    embedding_op_host((const uint32_t *)indices_host->data, params_host->data, out0_host->data, indices->shapes,
                      params->shapes, out0->shapes, indices->strides, params->strides, out0->strides, params->dtype);

    const void *p = out0->data.host;

    ck_assert_float_le(compare_buf(out0_host->data, p, out0->shapes, out0->strides, out0->bytes, out0->dtype), 1e-5);

    vkllm_tensor_free(context, out0);
    vkllm_tensor_free(context, indices);
    vkllm_tensor_free(context, params);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
    vkllm_array_u8_free(indices_host);
    vkllm_array_u8_free(params_host);
}
END_TEST;

Suite *vkllm_op_embedding_test_suit(void)
{
    Suite *suite = NULL;
    TCase *case_f32 = NULL;
    TCase *case_f16 = NULL;

    suite = suite_create("vkllm_op_embedding");
    case_f32 = tcase_create("vkllm_op_embedding_f32");
    tcase_add_loop_test(case_f32, test_embedding_op_f32, 0, 2);
    tcase_set_timeout(case_f32, 60.0);
    suite_add_tcase(suite, case_f32);

    case_f16 = tcase_create("vkllm_op_embedding_f16");
    tcase_add_loop_test(case_f16, test_embedding_op_f16, 0, 2);
    tcase_set_timeout(case_f16, 60.0);
    suite_add_tcase(suite, case_f16);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_embedding_test_suit();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
