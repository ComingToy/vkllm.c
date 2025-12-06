#ifndef __VKLLM_CONTEXT_H__
#define __VKLLM_CONTEXT_H__

#include <log.h>

#include "vkllm_errors.h"

struct vkllm_context {
  const char *appname;
};

extern vkllm_err_t vkllm_context_new(struct vkllm_context **context);
extern void vkllm_context_free(struct vkllm_context *pcontext);

#endif
