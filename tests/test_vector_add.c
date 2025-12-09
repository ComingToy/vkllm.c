#include "src/vkllm_commands.h"
#include "src/vkllm_common.h"
#include "src/vkllm_op_vector_add.h"
#include "src/vkllm_tensor.h"
#include <stdio.h>

static void random_buf(float *a, const size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = 10.0 * (rand() % 100) / 100.0;
    }
}

int main(void)
{
    struct vkllm_gpu_device *device;
    struct vkllm_context *context;
    vkllm_err_t err = vkllm_context_new(&context);
    if (err != VKLLM_ERR_OK)
    {
        return -1;
    }

    err = vkllm_gpu_device_new(context, 0, &device);
    if (err != VKLLM_ERR_OK)
    {
        return -1;
    }

    struct vkllm_commands *commands;
    err = vkllm_commands_new(context, device, &commands);
    if (err != VKLLM_ERR_OK)
    {
        return -1;
    }

    float buf[128] = {0};
    float buf1[128] = {0};
    random_buf(buf, 128);
    random_buf(buf1, 128);

    uint32_t shapes[] = {1, 1, 1, 128};
    struct vkllm_tensor *in0, *in1, *out0;
    vkllm_tensor_new(context, device, "in0", shapes, vkllm_float32, VKLLM_OP_ADD, NULL, 0, NULL, 0, false, &in0);
    vkllm_tensor_new(context, device, "in1", shapes, vkllm_float32, VKLLM_OP_ADD, NULL, 0, NULL, 0, false, &in1);

    struct vkllm_tensor *srcs[] = {in0, in1};
    vkllm_tensor_new(context, device, "out0", shapes, vkllm_float32, VKLLM_OP_ADD, srcs, 2, NULL, 0, true, &out0);

    _CHECK(vkllm_commands_begin(context, commands));
    _CHECK(vkllm_commands_upload(context, commands, in0, (const uint8_t *)buf, sizeof(buf)));
    _CHECK(vkllm_commands_upload(context, commands, in1, (const uint8_t *)buf1, sizeof(buf1)));
    _CHECK(vkllm_op_vector_add(context, commands, out0));
    _CHECK(vkllm_commands_end(context, commands));
    _CHECK(vkllm_commands_submit(context, commands));
    _CHECK(vkllm_commands_wait_exec(context, commands));
    _CHECK(vkllm_tensor_flush_cache(context, out0));

    const float *p = out0->data.host;
    fprintf(stderr, "vector add out: ");
    for (uint32_t i = 0; i < 128; ++i)
    {
        fprintf(stderr, "%f ", p[i]);
    }

    return 0;
}
