#ifndef __VKLLM_COMMON_H__
#define __VKLLM_COMMON_H__

#include <stddef.h>
#include <stdlib.h>
#include "vkllm_errors.h"

#define _NEW_AND_CHECK(_p, _type)             \
    do {                                      \
	(_p) = (_type*)malloc(sizeof(_type)); \
	if (!_p) return VKLLM_ERR_ALLOC;      \
    } while (0)

#endif
