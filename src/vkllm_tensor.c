#include "vkllm_tensor.h"

extern struct vkllm_tensor *
vkllm_new_tensor(struct vkllm_context *context, const uint32_t *shapes,
                 const uint32_t n_shape, struct vkllm_tensor *srcs,
                 const uint32_t n_srcs, void *params, bool mapped);
