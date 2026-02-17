#include "src/vkllm_commands.h"
#include "src/vkllm_gpu_device.h"
#include "src/vkllm_tensor.h"
#include <stdio.h>

int main(int argc, const char *argv[])
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

    struct vkllm_tensor *tensor;
    uint32_t shapes[] = {1, 1, 1, 256};
    vkllm_tensor_new(context, device, "t0", shapes, vkllm_float32, VKLLM_OP_ADD, NULL, 0, NULL, 0, false, &tensor);

    float buf[256] = {0};
    for (int i = 0; i < 256; ++i)
    {
        buf[i] = (float)i;
    }

    float buf1[256] = {0};

    vkllm_commands_begin(context, commands);
    vkllm_commands_upload(context, commands, tensor, (const uint8_t *)buf, sizeof(buf));
    vkllm_commands_download(context, commands, tensor, (uint8_t *)buf1, sizeof(buf1));
    vkllm_commands_end(context, commands);
    vkllm_commands_submit(context, commands);
    vkllm_commands_wait_exec(context, commands);

    fprintf(stderr, "download output: \n");
    for (int i = 0; i < 256; ++i)
    {
        fprintf(stderr, "%f ", buf1[i]);
    }
    fprintf(stderr, "\n");
    return 0;
}
