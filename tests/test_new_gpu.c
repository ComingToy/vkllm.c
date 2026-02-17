#include <log.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/vkllm_context.h"
#include "src/vkllm_errors.h"
#include "src/vkllm_gpu_device.h"

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "usage ./%s <gpu id>\n", argv[0]);
        return -1;
    }

    struct vkllm_context *context = NULL;
    vkllm_err_t e = vkllm_new_context(&context);
    if (e != VKLLM_ERR_OK)
    {
        log_error("vkllm_new_context failed: %s", vkllm_err_s(e));
        return -1;
    }

    int id = atoi(argv[1]);
    struct vkllm_gpu_device *gpu = NULL;
    e = vkllm_gpu_device_new(context, id, &gpu);
    if (e != VKLLM_ERR_OK)
    {
        log_error("vkllm_new_gpu_device failed: %s", vkllm_err_s(e));
        return -1;
    }

    vkllm_gpu_device_free(context, gpu);
    vkllm_destroy_context(context);
    return 0;
}
