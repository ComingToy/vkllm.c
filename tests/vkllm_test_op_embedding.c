#include "check.h"
#include "src/vkllm_array.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_dtypes.h"
#include "src/vkllm_op_add.h"
#include "src/vkllm_tensor.h"
#include "vkllm_test_common.h"
#

static struct
{
    uint32_t shapes0[4];
    uint32_t shapes1[4];
    vkllm_dtype_t dtype;
} tests[] = {
    {.shapes0 = {1, 2, 3, 4}, .shapes1 = {1, 2, 3, 4}, .dtype = vkllm_dtype_float32},
};

START_TEST(test_embedding_op)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
}
END_TEST;
