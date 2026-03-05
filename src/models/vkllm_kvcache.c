#include "vkllm_kvcache.h"
#include "../core/vkllm_common.h"
#include "../core/vkllm_context.h"
#include "../core/vkllm_graph.h"
#include "../core/vkllm_op_update_rows.h"
#include "../core/vkllm_tensor.h"
#include "src/core/vkllm_dtypes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

vkllm_err_t vkllm_kvcache_new(struct vkllm_context *context, uint32_t kcache_shape[4], uint32_t vcache_shape[4],
                              uint32_t layer_counts, struct vkllm_kvcache **kvcache)
{
    _NEW_AND_CHECK(*kvcache, struct vkllm_kvcache);
    struct vkllm_kvcache *p = *kvcache;
    p->kcaches = NULL;
    p->vcaches = NULL;
    p->layer_counts = layer_counts;
    _ASSIGN4(p->kcache_shape, kcache_shape);
    _ASSIGN4(p->vcache_shape, vcache_shape);

    vkllm_err_t err = VKLLM_ERR_OK;
    _CHECK_JUMP(vkllm_array_tensor_new(&p->kcaches, layer_counts), err, cleanup_malloc);
    _CHECK_JUMP(vkllm_array_tensor_new(&p->vcaches, layer_counts), err, cleanup_kcache);

    char name[128] = {0};

    for (uint32_t i = 0; i < layer_counts; ++i)
    {
        struct vkllm_tensor *kcache = NULL, *vcache = NULL;
        uint32_t offsets = 0;
        snprintf(name, sizeof(name), "kcache_%u", i);
        _CHECK_JUMP(vkllm_tensor_new(context, name, kcache_shape, vkllm_dtype_float16, VKLLM_OP_UPDATE_ROWS, NULL, 0,
                                     &offsets, sizeof(offsets), false, &kcache),
                    err, cleanup_vcache);
        vkllm_array_tensor_append(p->kcaches, kcache);

        snprintf(name, sizeof(name), "vcache_%u", i);
        _CHECK_JUMP(vkllm_tensor_new(context, name, vcache_shape, vkllm_dtype_float16, VKLLM_OP_UPDATE_ROWS, NULL, 0,
                                     &offsets, sizeof(offsets), false, &vcache),
                    err, cleanup_vcache);
        vkllm_array_tensor_append(p->vcaches, vcache);
    }

    return VKLLM_ERR_OK;

cleanup_vcache:
    for (uint32_t i = 0; i < p->vcaches->used_n; ++i)
    {
        vkllm_tensor_free(context, p->vcaches->data[i]);
    }
    vkllm_array_tensor_free(p->vcaches);
cleanup_kcache:
    for (uint32_t i = 0; i < p->kcaches->used_n; ++i)
    {
        vkllm_tensor_free(context, p->kcaches->data[i]);
    }
    vkllm_array_tensor_free(p->kcaches);
cleanup_malloc:
    free(p);
    return err;
}

vkllm_err_t vkllm_kvcache_update(struct vkllm_context *context, struct vkllm_kvcache *kvcache,
                                 struct vkllm_tensor **key, struct vkllm_tensor **value, uint32_t layer,
                                 uint32_t offset)
{
    _CHECK_ARGS(context && kvcache && key && value);
    struct vkllm_tensor *pkey = *key, *pvalue = *value;
    _CHECK_ARGS(pkey && pvalue);
    _CHECK_ARGS(layer < kvcache->kcaches->used_n && layer < kvcache->vcaches->used_n);

    struct vkllm_tensor *kcache = kvcache->kcaches->data[layer];

    _CHECK_ARGS(pkey->shapes[0] <= kcache->shapes[0]);
    _CHECK_ARGS(pkey->shapes[1] <= kcache->shapes[1]);
    _CHECK_ARGS(pkey->shapes[2] + offset <= kcache->shapes[2]);
    _CHECK_ARGS(pkey->shapes[3] <= kcache->shapes[3]);

    kcache->srcs[0] = pkey;
    memcpy(kcache->params, &offset, sizeof(offset));

    struct vkllm_tensor *vcache = kvcache->vcaches->data[layer];
    _CHECK_ARGS(pvalue->shapes[0] <= vcache->shapes[0]);
    _CHECK_ARGS(pvalue->shapes[1] <= vcache->shapes[1]);
    _CHECK_ARGS(pvalue->shapes[2] + offset <= vcache->shapes[2]);
    _CHECK_ARGS(pvalue->shapes[3] <= vcache->shapes[3]);

    vcache->srcs[0] = pvalue;
    memcpy(vcache->params, &offset, sizeof(offset));

    uint32_t b = pkey->shapes[0], c = pkey->shapes[1], h = pkey->shapes[2] + offset, w = pkey->shapes[3];

    // FIXME: multiple batch support
    uint32_t extents[4] = {b, c, h, w};

    _CHECK(vkllm_tensor_slice0(context, kcache, extents, key));
    _CHECK(vkllm_tensor_slice0(context, vcache, extents, value));

    return VKLLM_ERR_OK;
}

void vkllm_kvcache_free(struct vkllm_context *context, struct vkllm_kvcache *kvcache)
{
    struct vkllm_kvcache *p = kvcache;
    if (p->vcaches)
    {
        for (uint32_t i = 0; i < p->vcaches->used_n; ++i)
        {
            vkllm_tensor_free(context, p->vcaches->data[i]);
        }
        vkllm_array_tensor_free(p->vcaches);
    }
    if (p->kcaches)
    {
        for (uint32_t i = 0; i < p->kcaches->used_n; ++i)
        {
            vkllm_tensor_free(context, p->kcaches->data[i]);
        }
        vkllm_array_tensor_free(p->kcaches);
    }
    free(p);
}
