#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#include <log.h>
#include <stdint.h>

#include "vkllm_errors.h"

struct vkllm_gpu_device;

struct vkllm_context
{
    const char *appname;
    struct vkllm_gpu_device *device;
};

extern vkllm_err_t vkllm_context_new(uint32_t dev, struct vkllm_context **context);
extern void vkllm_context_free(struct vkllm_context *pcontext);

#endif
