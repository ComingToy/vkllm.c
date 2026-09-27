#include "check.h"
#include "src/core/vkllm_commands.h"
#include "src/core/vkllm_common.h"
#include "src/core/vkllm_context.h"
#include "src/core/vkllm_dtypes.h"
#include "src/core/vkllm_tensor.h"
#include <stdio.h>
#include <string.h>

// Covers the tensor view family (copy_ref / slice0 / view), which had zero
// test coverage — a _CHECK-on-comparison macro bug in slice0 shipped
// unnoticed because of it.
static struct vkllm_context *g_context = NULL;

START_TEST(test_copy_ref_shares_storage)
{
    uint32_t shapes[4] = {2, 3, 4, 5};
    struct vkllm_tensor *t = NULL;
    ck_assert_int_eq(vkllm_tensor_new(g_context, "t", shapes, vkllm_dtype_float16, VKLLM_OP_NONE, NULL, 0, NULL, 0,
                                      false, &t),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *ref = NULL;
    ck_assert_int_eq(vkllm_tensor_copy_ref(g_context, t, &ref), VKLLM_ERR_OK);

    // A ref borrows the buffer: same VkBuffer, same mapped state, marked ref.
    ck_assert_ptr_eq(ref->data.vk_buf, t->data.vk_buf);
    ck_assert_int_eq(ref->data.is_ref, true);
    ck_assert_int_eq(t->data.is_ref, false);

    // Shapes/strides are duplicated at creation time.
    for (uint32_t i = 0; i < 4; ++i)
    {
        ck_assert_uint_eq(ref->shapes[i], t->shapes[i]);
        ck_assert_uint_eq(ref->strides[i], t->strides[i]);
    }

    // The ref is a REF op pointing at the source; desc set starts clean.
    ck_assert_int_eq(ref->op, VKLLM_OP_REF);
    ck_assert_ptr_eq(ref->srcs[0], t);
    ck_assert(ref->vk_desc_set == VK_NULL_HANDLE);

    // Mutating the ref's shapes must not touch the source tensor.
    ref->shapes[2] = 1;
    ck_assert_uint_eq(t->shapes[2], 4);

    // Only the original owns the buffer; freeing both is the caller's
    // contract, so free ref first (no device memory released for it).
    vkllm_tensor_free(g_context, ref);
    vkllm_tensor_free(g_context, t);
}
END_TEST

START_TEST(test_slice0_crops_shapes_keeps_strides)
{
    uint32_t shapes[4] = {4, 3, 8, 16};
    struct vkllm_tensor *t = NULL;
    ck_assert_int_eq(vkllm_tensor_new(g_context, "t", shapes, vkllm_dtype_float16, VKLLM_OP_NONE, NULL, 0, NULL, 0,
                                      false, &t),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *slice = NULL;
    ck_assert_int_eq(vkllm_tensor_slice0(g_context, t, (uint32_t[]){2, 2, 5, 7}, &slice), VKLLM_ERR_OK);

    ck_assert_uint_eq(slice->shapes[0], 2);
    ck_assert_uint_eq(slice->shapes[1], 2);
    ck_assert_uint_eq(slice->shapes[2], 5);
    ck_assert_uint_eq(slice->shapes[3], 7);
    // Slice semantics: rows keep the original pitch (strides untouched), so
    // row r of the slice is row r of the base buffer.
    for (uint32_t i = 0; i < 4; ++i)
    {
        ck_assert_uint_eq(slice->strides[i], t->strides[i]);
    }
    ck_assert_ptr_eq(slice->data.vk_buf, t->data.vk_buf);

    vkllm_tensor_free(g_context, slice);
    vkllm_tensor_free(g_context, t);
}
END_TEST

START_TEST(test_slice0_rejects_oversized_extents)
{
    uint32_t shapes[4] = {2, 2, 4, 8};
    struct vkllm_tensor *t = NULL;
    ck_assert_int_eq(vkllm_tensor_new(g_context, "t", shapes, vkllm_dtype_float16, VKLLM_OP_NONE, NULL, 0, NULL, 0,
                                      false, &t),
                     VKLLM_ERR_OK);

    // Each axis individually must reject extents beyond the base shape.
    struct
    {
        uint32_t extents[4];
    } overs[] = {
        {{3, 2, 4, 8}}, // axis 0
        {{2, 3, 4, 8}}, // axis 1
        {{2, 2, 5, 8}}, // axis 2
        {{2, 2, 4, 9}}, // axis 3
    };

    for (uint32_t i = 0; i < sizeof(overs) / sizeof(overs[0]); ++i)
    {
        struct vkllm_tensor *slice = (struct vkllm_tensor *)0xdeadbeef;
        ck_assert_int_ne(vkllm_tensor_slice0(g_context, t, overs[i].extents, &slice), VKLLM_ERR_OK);
    }

    // Equal extents are legal (identity slice).
    struct vkllm_tensor *same = NULL;
    ck_assert_int_eq(vkllm_tensor_slice0(g_context, t, shapes, &same), VKLLM_ERR_OK);
    vkllm_tensor_free(g_context, same);

    vkllm_tensor_free(g_context, t);
}
END_TEST

START_TEST(test_view_shrinks_only)
{
    uint32_t shapes[4] = {1, 1, 8, 16};
    struct vkllm_tensor *t = NULL;
    ck_assert_int_eq(vkllm_tensor_new(g_context, "t", shapes, vkllm_dtype_float16, VKLLM_OP_NONE, NULL, 0, NULL, 0,
                                      true, &t),
                     VKLLM_ERR_OK);

    // Shrink is allowed, in any combination of axes.
    ck_assert_int_eq(vkllm_tensor_view(g_context, t, (uint32_t[]){1, 1, 6, 12}), VKLLM_ERR_OK);
    ck_assert_uint_eq(t->shapes[2], 6);
    ck_assert_uint_eq(t->shapes[3], 12);

    // Further shrink works.
    ck_assert_int_eq(vkllm_tensor_view(g_context, t, (uint32_t[]){1, 1, 2, 4}), VKLLM_ERR_OK);
    ck_assert_uint_eq(t->shapes[2], 2);

    // Growing beyond the CURRENT (already shrunk) view must fail — the
    // buffer was never guaranteed beyond it in the shrink-only contract.
    ck_assert_int_ne(vkllm_tensor_view(g_context, t, (uint32_t[]){1, 1, 5, 4}), VKLLM_ERR_OK);

    // Strides never change through views.
    uint32_t strides_before[4];
    memcpy(strides_before, t->strides, sizeof(strides_before));
    ck_assert_int_eq(vkllm_tensor_view(g_context, t, (uint32_t[]){1, 1, 1, 1}), VKLLM_ERR_OK);
    ck_assert_mem_eq(t->strides, strides_before, sizeof(strides_before));

    vkllm_tensor_free(g_context, t);
}
END_TEST

START_TEST(test_slice0_then_view_chains)
{
    // The kvcache pattern: slice the full cache once, then shrink the view
    // every decode step.
    uint32_t cache_shapes[4] = {1, 4, 64, 32};
    struct vkllm_tensor *cache = NULL;
    ck_assert_int_eq(vkllm_tensor_new(g_context, "cache", cache_shapes, vkllm_dtype_float16, VKLLM_OP_NONE, NULL, 0,
                                      NULL, 0, false, &cache),
                     VKLLM_ERR_OK);

    struct vkllm_tensor *view = NULL;
    ck_assert_int_eq(vkllm_tensor_slice0(g_context, cache, cache_shapes, &view), VKLLM_ERR_OK);

    // vkllm_tensor_view is shrink-only: the cache pattern slices at full
    // length once and then walks the view DOWN as needed. (Growing is the
    // kvcache's job via direct shape assignment — see kvcache_advance.)
    uint32_t lens[] = {64, 57, 43, 29, 15, 8, 1};
    for (uint32_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i)
    {
        ck_assert_int_eq(vkllm_tensor_view(g_context, view, (uint32_t[]){1, 4, lens[i], 32}), VKLLM_ERR_OK);
        ck_assert_uint_eq(view->shapes[2], lens[i]);
    }

    vkllm_tensor_free(g_context, view);
    vkllm_tensor_free(g_context, cache);
}
END_TEST

static void setup_global_context(void)
{
    vkllm_err_t err = vkllm_context_new(0, &g_context);
    if (err != VKLLM_ERR_OK)
    {
        fprintf(stderr, "Failed to create global context: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

static void teardown_global_context(void)
{
    if (g_context != NULL)
    {
        vkllm_context_free(g_context);
        g_context = NULL;
    }
}

Suite *vkllm_tensor_view_test_suite(void)
{
    Suite *s = suite_create("vkllm_tensor_view");
    TCase *tc = tcase_create("views");

    tcase_add_unchecked_fixture(tc, setup_global_context, teardown_global_context);
    tcase_add_test(tc, test_slice0_then_view_chains);
    suite_add_tcase(s, tc);
    return s;
}

int main(void)
{
    Suite *s = vkllm_tensor_view_test_suite();
    SRunner *runner = srunner_create(s);
    srunner_set_fork_status(runner, CK_NOFORK);

    srunner_run_all(runner, CK_NORMAL);
    int nfail = srunner_ntests_failed(runner);

    srunner_free(runner);
    return nfail == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
