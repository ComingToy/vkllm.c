#include "log.h"
#include "src/vkllm_commands.h"
#include "src/vkllm_context.h"
#include "src/vkllm_gpu_device.h"
#include "src/vkllm_tensor.h"
#include <stdio.h>

void random_buf(float *a, const size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = 10.0 * (rand() % 100) / 100.0;
    }
}

float float_mse(const float *a, const float *b, const size_t n)
{
    float err = .0f;
    float w = 1.0f / n;
    for (size_t i = 0; i < n; ++i)
    {
        err = err + w * (a[i] - b[i]) * (a[i] - b[i]);
    }

    return err;
}

int main(int argc, const char *argv[])
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

    float buf[65536] = {0};
    float buf1[65536] = {.0};
    struct vkllm_tensor *tensor;
    uint32_t shapes[] = {1, 1, 1, 65536};
    vkllm_tensor_new(context, "t0", shapes, vkllm_dtype_float32, VKLLM_OP_ADD, NULL, 0, NULL, 0, false, &tensor);

    random_buf(buf, 65536);

    float mse = float_mse(buf, buf1, 65536);
    log_info("mse before transfer: %f", mse);
    vkllm_commands_begin(context, commands);
    vkllm_commands_upload(context, commands, tensor, (const uint8_t *)buf, sizeof(buf));
    vkllm_commands_download(context, commands, tensor, (uint8_t *)buf1, sizeof(buf1));
    vkllm_commands_end(context, commands);
    vkllm_commands_submit(context, commands);
    vkllm_commands_wait_exec(context, commands);

    mse = float_mse(buf, buf1, 65536);
    log_info("mse after transfer: %f", mse);

    vkllm_tensor_free(context, tensor);
    vkllm_commands_free(context, commands);
    vkllm_context_free(context);
    return 0;
}
