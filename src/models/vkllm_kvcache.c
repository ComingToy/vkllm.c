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
    p->kviews = NULL;
    p->vviews = NULL;
    p->layer_counts = layer_counts;
    p->cur_len = 0;
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

static vkllm_err_t kvcache_make_views(struct vkllm_context *context, struct vkllm_kvcache *kvcache)
{
    if (kvcache->kviews)
        return VKLLM_ERR_OK;

    vkllm_err_t err = vkllm_array_tensor_new(&kvcache->kviews, kvcache->layer_counts);
    if (err != VKLLM_ERR_OK)
    {
        log_error("make_views: kviews array new failed: %s", vkllm_err_s(err));
        return err;
    }

    err = vkllm_array_tensor_new(&kvcache->vviews, kvcache->layer_counts);
    if (err != VKLLM_ERR_OK)
    {
        log_error("make_views: vviews array new failed: %s", vkllm_err_s(err));
        return err;
    }

    for (uint32_t i = 0; i < kvcache->layer_counts; ++i)
    {
        struct vkllm_tensor *kview = NULL, *vview = NULL;
        // Views start at FULL context length so that any later logical
        // length (prefill or any decode step) is a shrink — tensor_view
        // only allows shrinking.
        uint32_t extents[4];
        _ASSIGN4(extents, kvcache->kcache_shape);
        err = vkllm_tensor_slice0(context, kvcache->kcaches->data[i], extents, &kview);
        if (err != VKLLM_ERR_OK)
        {
            log_error("make_views: k slice %u failed: %s", i, vkllm_err_s(err));
            return err;
        }
        vkllm_array_tensor_append(kvcache->kviews, kview);

        _ASSIGN4(extents, kvcache->vcache_shape);
        err = vkllm_tensor_slice0(context, kvcache->vcaches->data[i], extents, &vview);
        if (err != VKLLM_ERR_OK)
        {
            log_error("make_views: v slice %u failed: %s", i, vkllm_err_s(err));
            return err;
        }
        vkllm_array_tensor_append(kvcache->vviews, vview);
    }

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_kvcache_bind(struct vkllm_context *context, struct vkllm_kvcache *kvcache,
                               struct vkllm_graph *graph, struct vkllm_tensor **key, struct vkllm_tensor **value,
                               uint32_t layer, uint32_t offset)
{
    _CHECK_ARGS(context && kvcache && graph && key && value);
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

    // Lazily create the persistent views once, then only re-view them.
    _CHECK(kvcache_make_views(context, kvcache));

    // Cache tensors run as part of the graph but stay owned by the model.
    _CHECK(vkllm_graph_add_external(context, graph, kcache));
    _CHECK(vkllm_graph_add_external(context, graph, vcache));

    uint32_t cur_len = offset + pkey->shapes[2];
    // Views may be re-set to any length within the full cache (views created
    // by a previous graph build may have been shrunk below the new length).
    uint32_t kext[4] = {pkey->shapes[0], pkey->shapes[1], cur_len, pkey->shapes[3]};
    uint32_t vext[4] = {pvalue->shapes[0], pvalue->shapes[1], cur_len, pvalue->shapes[3]};
    kvcache->kviews->data[layer]->shapes[2] = kext[2];
    kvcache->vviews->data[layer]->shapes[2] = vext[2];
    kvcache->cur_len = cur_len;

    *key = kvcache->kviews->data[layer];
    *value = kvcache->vviews->data[layer];

    return VKLLM_ERR_OK;
}

vkllm_err_t vkllm_kvcache_advance(struct vkllm_context *context, struct vkllm_kvcache *kvcache, uint32_t offset,
                                  uint32_t seq_len)
{
    _CHECK_ARGS(context && kvcache);
    _CHECK_ARGS(kvcache->kviews && kvcache->vviews);
    _CHECK_ARGS(offset + seq_len <= kvcache->kcache_shape[2]);

    for (uint32_t i = 0; i < kvcache->layer_counts; ++i)
    {
        // Move the update-rows write offset for this step.
        memcpy(kvcache->kcaches->data[i]->params, &offset, sizeof(offset));
        memcpy(kvcache->vcaches->data[i]->params, &offset, sizeof(offset));

        // Grow the attention views to cover [0, offset + seq_len). Direct
        // shape assignment: the view may currently be shorter (previous
        // step) and tensor_view only allows shrinking; the capacity check
        // is offset + seq_len <= kcache_shape[2] above.
        uint32_t cur_len = offset + seq_len;
        kvcache->kviews->data[i]->shapes[2] = cur_len;
        kvcache->vviews->data[i]->shapes[2] = cur_len;
    }

    kvcache->cur_len = offset + seq_len;
    return VKLLM_ERR_OK;
}

void vkllm_kvcache_free(struct vkllm_context *context, struct vkllm_kvcache *kvcache)
{
    struct vkllm_kvcache *p = kvcache;
    if (p->vviews)
    {
        for (uint32_t i = 0; i < p->vviews->used_n; ++i)
        {
            vkllm_tensor_free(context, p->vviews->data[i]);
        }
        vkllm_array_tensor_free(p->vviews);
    }
    if (p->kviews)
    {
        for (uint32_t i = 0; i < p->kviews->used_n; ++i)
        {
            vkllm_tensor_free(context, p->kviews->data[i]);
        }
        vkllm_array_tensor_free(p->kviews);
    }
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
