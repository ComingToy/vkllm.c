#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_add.h"
#include "src/vkllm_op_embedding.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#

static struct
{
    uint32_t shapes0[4];
    uint32_t shapes1[4];
    vkllm_dtype_t dtype;
} tests[] = {
    {.shapes0 = {1, 1, 32, 32}, .shapes1 = {1, 1, 128, 64}, .dtype = vkllm_dtype_float32},
    {.shapes0 = {1, 1, 32, 99}, .shapes1 = {1, 1, 128, 64}, .dtype = vkllm_dtype_float32},
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

    _LOOP_SHAPE(shapes, {
        uint32_t i = get_indice(_b, _c, _h, _w, strides, sizeof(uint32_t));
        uint32_t tok = indices[i];
        if (tok >= shapes1[2])
        {
            tok = 0;
        }

        uint32_t params_off = get_indice(1, 1, tok, 0, strides1, dsize);
        uint32_t out_off = get_indice(_c, _h, _w, 0, strides2, dsize);

        for (uint32_t k = 0; k < shapes1[3]; ++k)
        {
            if (dtype == vkllm_dtype_float32)
            {
                out_fp32[out_off + k] = params_fp32[params_off + k];
            }
        }
    });
}

START_TEST(test_embedding_op)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *indices = NULL, *params = NULL;
    ck_assert_int_eq(vkllm_tensor_new(context, "indices", tests[_i].shapes0, vkllm_dtype_uint32, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &indices),
                     VKLLM_ERR_OK);

    ck_assert_int_eq(vkllm_tensor_new(context, "params", tests[_i].shapes1, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &params),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *out0 = NULL;
    uint32_t shapes_out[] = {tests[_i].shapes0[0], tests[_i].shapes0[1], tests[_i].shapes0[2], tests[_i].shapes1[3]};
    struct vkllm_tensor *srcs[] = {indices, params};

    uint32_t UNK_TOK = 0;
    ck_assert_int_eq(vkllm_tensor_new(context, "out0", shapes_out, vkllm_dtype_float32, VKLLM_OP_EMBEDDING, srcs, 2,
                                      (const uint8_t *)&UNK_TOK, sizeof(UNK_TOK), true, &out0),
                     VKLLM_ERR_OK);

    struct vkllm_array_u8 *indices_host = NULL, *params_host = NULL, *out0_host = NULL;

    vkllm_array_u8_new(&indices_host, indices->bytes);
    vkllm_array_u8_new(&params_host, params->bytes);
    vkllm_array_u8_new(&out0_host, out0->bytes);

    random_tensor(indices_host, indices->shapes, indices->strides, indices->dtype);
    random_tensor(params_host, params->shapes, params->strides, params->dtype);

    vkllm_commands_begin(context, commands);
    vkllm_commands_upload(context, commands, indices, indices_host->data, indices_host->alloc_n);
    vkllm_commands_upload(context, commands, params, params_host->data, params_host->alloc_n);
    vkllm_op_embedding(context, commands, out0);
    vkllm_commands_end(context, commands);
    vkllm_commands_submit(context, commands);
    vkllm_commands_wait_exec(context, commands);
    vkllm_tensor_flush_cache(context, out0);

    embedding_op_host((const uint32_t *)indices_host->data, params_host->data, out0_host->data, indices->shapes,
                      params->shapes, out0->shapes, indices->strides, params->strides, out0->strides, params->dtype);

    const float *p = out0->data.host;
    ck_assert_float_le(compare_buf(out0_host->data, p, out0->shapes, out0->strides, out0->bytes, out0->dtype), 1e-5);

    print_n("out0 host", (const float *)out0_host->data, 10);
    print_n("out0", p, 10);

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

    suite = suite_create("vkllm_op_embedding");
    case_f32 = tcase_create("vkllm_op_embedding_f32");
    tcase_add_loop_test(case_f32, test_embedding_op, 0, 2);
    tcase_set_timeout(case_f32, 60.0);
    suite_add_tcase(suite, case_f32);
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
