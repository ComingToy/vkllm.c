#include "src/vkllm_commands.h"
#include "src/vkllm_common.h"
#include "src/vkllm_context.h"
#include "src/vkllm_op_add.h"
#include "src/vkllm_tensor.h"
#include <stdio.h>

static void random_buf(float *a, const size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = 10.0 * (rand() % 100) / 100.0;
    }
}

static void add_buf(const float* a, const float* b, float* c, size_t n)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

static float compare_buf(const float* a, const float* b, size_t n)
{
    float w = 1.0 / n;
    float err = .0;
    for (uint32_t i = 0; i < n; ++i)
    {
        err = err + w * (a[i] - b[i]) * (a[i] - b[i]);
    }

    return err;
}

static void print_vec(const char *name, const float *a, const size_t n)
{
    fprintf(stderr, "%s: ", name);
    for (size_t i = 0; i < n; ++i)
    {
        fprintf(stderr, "%f ", a[i]);
    }
    fprintf(stderr, "\n");
}

int main(void)
{
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(0, &context);
    if (err != VKLLM_ERR_OK)
    {
        return -1;
    }

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, &commands);
    if (err != VKLLM_ERR_OK)
    {
        return -1;
    }

#define n (3 * 4 * 5 * 128)
    float buf[n] = {0};
    float buf1[n] = {0};
    float buf2[n] = {0};
    random_buf(buf, n);
    random_buf(buf1, n);
    add_buf(buf, buf1, buf2, n);

    uint32_t shapes[] = {3, 4, 5, 128};
    struct vkllm_tensor *in0, *in1, *out0;
    vkllm_tensor_new(context, "in0", shapes, vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, &in0);
    vkllm_tensor_new(context, "in1", shapes, vkllm_dtype_float32, VKLLM_OP_NONE, NULL, 0, NULL, 0, false, &in1);

    struct vkllm_tensor *srcs[] = {in0, in1};
    vkllm_tensor_new(context, "out0", shapes, vkllm_dtype_float32, VKLLM_OP_ADD, srcs, 2, NULL, 0, true, &out0);

    // print_vec("in0: ", buf, 3 * 4 * 5 * 128);
    // print_vec("in1: ", buf1, 3 * 4 * 5 * 128);
    _CHECK(vkllm_commands_begin(context, commands));
    _CHECK(vkllm_commands_upload(context, commands, in0, (const uint8_t *)buf, sizeof(buf)));
    _CHECK(vkllm_commands_upload(context, commands, in1, (const uint8_t *)buf1, sizeof(buf1)));
    _CHECK(vkllm_op_add(context, commands, out0));
    _CHECK(vkllm_commands_end(context, commands));
    _CHECK(vkllm_commands_submit(context, commands));
    _CHECK(vkllm_commands_wait_exec(context, commands));
    _CHECK(vkllm_tensor_flush_cache(context, out0));
    _CHECK(vkllm_tensor_invalid_cache(context, out0));

    const float *p = out0->data.host;
    // print_vec("vec add out: ", p, n);
    log_info("add mse = %f", compare_buf(buf2, p, n));

    vkllm_tensor_free(context, in0);
    vkllm_tensor_free(context, in1);
    vkllm_tensor_free(context, out0);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
    return 0;
}
