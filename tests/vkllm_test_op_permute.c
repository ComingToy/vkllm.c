#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#include <stdio.h>
#include <string.h>

static struct
{
    uint32_t shapes[4];
    uint32_t axis[4];
    vkllm_dtype_t dtype;
} tests[] = {
    // Test various permutations with float32
    {{2, 3, 4, 5}, {0, 1, 2, 3}, vkllm_dtype_float32},   // Identity permutation
    {{2, 3, 4, 5}, {0, 1, 3, 2}, vkllm_dtype_float32},   // Swap last two dimensions
    {{2, 3, 4, 5}, {0, 2, 1, 3}, vkllm_dtype_float32},   // Swap middle two dimensions
    {{2, 3, 4, 5}, {3, 2, 1, 0}, vkllm_dtype_float32},   // Reverse all dimensions
    {{4, 8, 16, 32}, {1, 0, 2, 3}, vkllm_dtype_float32}, // Swap first two dimensions
    {{3, 5, 7, 9}, {2, 0, 3, 1}, vkllm_dtype_float32},   // Complex permutation
    {{2, 4, 6, 8}, {3, 1, 2, 0}, vkllm_dtype_float32},   // Another complex permutation
    
    // Test with float16
    {{2, 3, 4, 5}, {0, 1, 3, 2}, vkllm_dtype_float16},   // Swap last two dimensions
    {{4, 8, 16, 32}, {1, 0, 2, 3}, vkllm_dtype_float16}, // Swap first two dimensions
    {{3, 5, 7, 9}, {2, 0, 3, 1}, vkllm_dtype_float16},   // Complex permutation
};

START_TEST(test_op_permute)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_tensor *tensor;
    
    // Create a tensor with initial data
    ck_assert_int_eq(vkllm_tensor_new(context, "tensor", tests[_i].shapes, tests[_i].dtype, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, true, &tensor),
                     VKLLM_ERR_OK);

    struct vkllm_dtype_info dtype_info;
    vkllm_get_dtype_info(tests[_i].dtype, &dtype_info);

    // Allocate buffers
    void *input_buf = malloc(tensor->bytes);
    ck_assert_ptr_nonnull(input_buf);

    // Fill input buffer with random data
    random_tensor(input_buf, tensor->shapes, tensor->strides, tests[_i].dtype);

    // Copy input data to tensor
    memcpy(tensor->data.host, input_buf, tensor->bytes);
    
    // Store original shapes and strides
    uint32_t orig_shapes[4] = {tensor->shapes[0], tensor->shapes[1], tensor->shapes[2], tensor->shapes[3]};
    uint32_t orig_strides[4] = {tensor->strides[0], tensor->strides[1], tensor->strides[2], tensor->strides[3]};

    // Perform permute operation
    ck_assert_int_eq(vkllm_tensor_permute(context, tensor, tests[_i].axis), VKLLM_ERR_OK);

    // Calculate expected shapes and strides after permutation
    uint32_t expected_shapes[4] = {orig_shapes[tests[_i].axis[0]], orig_shapes[tests[_i].axis[1]],
                                   orig_shapes[tests[_i].axis[2]], orig_shapes[tests[_i].axis[3]]};
    uint32_t expected_strides[4] = {orig_strides[tests[_i].axis[0]], orig_strides[tests[_i].axis[1]],
                                    orig_strides[tests[_i].axis[2]], orig_strides[tests[_i].axis[3]]};

    // Verify shapes are updated correctly
    ck_assert_uint_eq(tensor->shapes[0], expected_shapes[0]);
    ck_assert_uint_eq(tensor->shapes[1], expected_shapes[1]);
    ck_assert_uint_eq(tensor->shapes[2], expected_shapes[2]);
    ck_assert_uint_eq(tensor->shapes[3], expected_shapes[3]);

    // Verify strides are updated correctly
    ck_assert_uint_eq(tensor->strides[0], expected_strides[0]);
    ck_assert_uint_eq(tensor->strides[1], expected_strides[1]);
    ck_assert_uint_eq(tensor->strides[2], expected_strides[2]);
    ck_assert_uint_eq(tensor->strides[3], expected_strides[3]);

    // Now verify element-by-element data access
    // For each logical position in the permuted tensor, verify it accesses the correct original data
    uint32_t orig_es[4] = {orig_strides[0] / dtype_info.bytes, orig_strides[1] / dtype_info.bytes,
                           orig_strides[2] / dtype_info.bytes, orig_strides[3] / dtype_info.bytes};
    
    uint32_t new_es[4] = {tensor->strides[0] / dtype_info.bytes, tensor->strides[1] / dtype_info.bytes,
                          tensor->strides[2] / dtype_info.bytes, tensor->strides[3] / dtype_info.bytes};

    uint32_t num_errors = 0;
    const uint32_t max_errors_to_print = 10;

    if (tests[_i].dtype == vkllm_dtype_float16)
    {
        const vkllm_fp16_pack *orig_data = (const vkllm_fp16_pack *)input_buf;
        const vkllm_fp16_pack *perm_data = (const vkllm_fp16_pack *)tensor->data.host;

        // Iterate through all logical positions in the permuted tensor
        for (uint32_t i0 = 0; i0 < tensor->shapes[0]; ++i0)
        {
            for (uint32_t i1 = 0; i1 < tensor->shapes[1]; ++i1)
            {
                for (uint32_t i2 = 0; i2 < tensor->shapes[2]; ++i2)
                {
                    for (uint32_t i3 = 0; i3 < tensor->shapes[3]; ++i3)
                    {
                        // Calculate physical index in permuted view
                        uint32_t perm_idx = i0 * new_es[0] + i1 * new_es[1] + i2 * new_es[2] + i3 * new_es[3];
                        
                        // Calculate which original position this should map to
                        // If axis = [1,0,2,3], then new[i0,i1,i2,i3] should equal old[i1,i0,i2,i3]
                        uint32_t orig_indices[4] = {i0, i1, i2, i3};
                        uint32_t mapped_indices[4];
                        for (int k = 0; k < 4; k++)
                        {
                            mapped_indices[tests[_i].axis[k]] = orig_indices[k];
                        }
                        
                        uint32_t orig_idx = mapped_indices[0] * orig_es[0] + mapped_indices[1] * orig_es[1] +
                                           mapped_indices[2] * orig_es[2] + mapped_indices[3] * orig_es[3];
                        
                        // Compare values
                        float perm_val = vkllm_fp16_to_fp32(perm_data[perm_idx]);
                        float orig_val = vkllm_fp16_to_fp32(orig_data[orig_idx]);
                        
                        if (fabsf(perm_val - orig_val) > 1e-5)
                        {
                            if (num_errors < max_errors_to_print)
                            {
                                printf("Error at permuted position [%u,%u,%u,%u] (maps to orig [%u,%u,%u,%u]): "
                                       "expected %f, got %f (indices: perm=%u, orig=%u)\n",
                                       i0, i1, i2, i3, mapped_indices[0], mapped_indices[1], mapped_indices[2],
                                       mapped_indices[3], orig_val, perm_val, perm_idx, orig_idx);
                            }
                            num_errors++;
                        }
                    }
                }
            }
        }
    }
    else // float32
    {
        const float *orig_data = (const float *)input_buf;
        const float *perm_data = (const float *)tensor->data.host;

        // Iterate through all logical positions in the permuted tensor
        for (uint32_t i0 = 0; i0 < tensor->shapes[0]; ++i0)
        {
            for (uint32_t i1 = 0; i1 < tensor->shapes[1]; ++i1)
            {
                for (uint32_t i2 = 0; i2 < tensor->shapes[2]; ++i2)
                {
                    for (uint32_t i3 = 0; i3 < tensor->shapes[3]; ++i3)
                    {
                        // Calculate physical index in permuted view
                        uint32_t perm_idx = i0 * new_es[0] + i1 * new_es[1] + i2 * new_es[2] + i3 * new_es[3];
                        
                        // Calculate which original position this should map to
                        uint32_t orig_indices[4] = {i0, i1, i2, i3};
                        uint32_t mapped_indices[4];
                        for (int k = 0; k < 4; k++)
                        {
                            mapped_indices[tests[_i].axis[k]] = orig_indices[k];
                        }
                        
                        uint32_t orig_idx = mapped_indices[0] * orig_es[0] + mapped_indices[1] * orig_es[1] +
                                           mapped_indices[2] * orig_es[2] + mapped_indices[3] * orig_es[3];
                        
                        // Compare values
                        float perm_val = perm_data[perm_idx];
                        float orig_val = orig_data[orig_idx];
                        
                        if (fabsf(perm_val - orig_val) > 1e-5)
                        {
                            if (num_errors < max_errors_to_print)
                            {
                                printf("Error at permuted position [%u,%u,%u,%u] (maps to orig [%u,%u,%u,%u]): "
                                       "expected %f, got %f (indices: perm=%u, orig=%u)\n",
                                       i0, i1, i2, i3, mapped_indices[0], mapped_indices[1], mapped_indices[2],
                                       mapped_indices[3], orig_val, perm_val, perm_idx, orig_idx);
                            }
                            num_errors++;
                        }
                    }
                }
            }
        }
    }

    if (num_errors > 0)
    {
        printf("Total errors: %u out of %u elements\n", num_errors, _MUL4(tensor->shapes));
    }
    ck_assert_uint_eq(num_errors, 0);

    printf("Test %d passed: shapes [%u,%u,%u,%u] with axis [%u,%u,%u,%u] -> shapes [%u,%u,%u,%u], all %u elements verified\n",
           _i, orig_shapes[0], orig_shapes[1], orig_shapes[2], orig_shapes[3], tests[_i].axis[0], tests[_i].axis[1],
           tests[_i].axis[2], tests[_i].axis[3], tensor->shapes[0], tensor->shapes[1], tensor->shapes[2],
           tensor->shapes[3], _MUL4(tensor->shapes));

    // Clean up
    free(input_buf);
    vkllm_tensor_free(context, tensor);
    vkllm_context_free(context);
}
END_TEST;

Suite *vkllm_op_permute_test_suite(void)
{
    Suite *suite = NULL;
    TCase *tcase_f32 = NULL, *tcase_f16 = NULL;
    suite = suite_create("vkllm_op_permute");
    tcase_f32 = tcase_create("vkllm_op_permute_f32");
    tcase_f16 = tcase_create("vkllm_op_permute_f16");

    tcase_add_loop_test(tcase_f32, test_op_permute, 0, 7);
    tcase_add_loop_test(tcase_f16, test_op_permute, 7, 10);
    tcase_set_timeout(tcase_f32, 60.0);
    tcase_set_timeout(tcase_f16, 60.0);
    suite_add_tcase(suite, tcase_f32);
    suite_add_tcase(suite, tcase_f16);
    return suite;
}

int main(void)
{
    Suite *s = vkllm_op_permute_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
