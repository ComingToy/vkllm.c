#include "check.h"
#include "src/vkllm_hashset.h"
#include <stdio.h>

START_TEST(test_hashset_create_and_free)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_ptr_nonnull(set);
    ck_assert_int_eq(vkllm_hashset_size(set), 0);
    ck_assert(vkllm_hashset_empty(set));
    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_insert_and_contains)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Insert some keys
    err = vkllm_hashset_insert(set, 42);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    err = vkllm_hashset_insert(set, 123);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    err = vkllm_hashset_insert(set, 999);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Check size
    ck_assert_int_eq(vkllm_hashset_size(set), 3);
    ck_assert(!vkllm_hashset_empty(set));

    // Check contains
    ck_assert(vkllm_hashset_contains(set, 42));
    ck_assert(vkllm_hashset_contains(set, 123));
    ck_assert(vkllm_hashset_contains(set, 999));
    ck_assert(!vkllm_hashset_contains(set, 1));
    ck_assert(!vkllm_hashset_contains(set, 100));

    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_insert_duplicate)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Insert key
    err = vkllm_hashset_insert(set, 42);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashset_size(set), 1);

    // Insert same key again
    err = vkllm_hashset_insert(set, 42);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashset_size(set), 1); // Size should not change

    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_remove)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Insert keys
    vkllm_hashset_insert(set, 10);
    vkllm_hashset_insert(set, 20);
    vkllm_hashset_insert(set, 30);
    ck_assert_int_eq(vkllm_hashset_size(set), 3);

    // Remove a key
    err = vkllm_hashset_remove(set, 20);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashset_size(set), 2);
    ck_assert(!vkllm_hashset_contains(set, 20));
    ck_assert(vkllm_hashset_contains(set, 10));
    ck_assert(vkllm_hashset_contains(set, 30));

    // Remove non-existent key (should not fail)
    err = vkllm_hashset_remove(set, 999);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashset_size(set), 2);

    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_clear)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Insert keys
    vkllm_hashset_insert(set, 1);
    vkllm_hashset_insert(set, 2);
    vkllm_hashset_insert(set, 3);
    ck_assert_int_eq(vkllm_hashset_size(set), 3);

    // Clear the set
    vkllm_hashset_clear(set);
    ck_assert_int_eq(vkllm_hashset_size(set), 0);
    ck_assert(vkllm_hashset_empty(set));
    ck_assert(!vkllm_hashset_contains(set, 1));
    ck_assert(!vkllm_hashset_contains(set, 2));
    ck_assert(!vkllm_hashset_contains(set, 3));

    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_resize)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 4); // Small initial capacity
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Insert many keys to trigger resize
    for (uint64_t i = 0; i < 100; i++)
    {
        err = vkllm_hashset_insert(set, i);
        ck_assert_int_eq(err, VKLLM_ERR_OK);
    }

    ck_assert_int_eq(vkllm_hashset_size(set), 100);

    // Verify all keys exist
    for (uint64_t i = 0; i < 100; i++)
    {
        ck_assert_msg(vkllm_hashset_contains(set, i), "Key %lu should exist", i);
    }

    // Verify non-existent keys
    ck_assert(!vkllm_hashset_contains(set, 200));
    ck_assert(!vkllm_hashset_contains(set, 300));

    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_large_keys)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Test with large uint64_t values
    uint64_t large_keys[] = {
        0xFFFFFFFFFFFFFFFFULL,
        0x8000000000000000ULL,
        0x123456789ABCDEFULL,
        0xDEADBEEFCAFEBABEULL,
        0x0000000000000001ULL
    };

    for (size_t i = 0; i < sizeof(large_keys) / sizeof(large_keys[0]); i++)
    {
        err = vkllm_hashset_insert(set, large_keys[i]);
        ck_assert_int_eq(err, VKLLM_ERR_OK);
    }

    for (size_t i = 0; i < sizeof(large_keys) / sizeof(large_keys[0]); i++)
    {
        ck_assert(vkllm_hashset_contains(set, large_keys[i]));
    }

    vkllm_hashset_free(set);
}
END_TEST

START_TEST(test_hashset_collision_handling)
{
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, 8); // Small capacity for more collisions
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    // Insert keys that may collide
    for (uint64_t i = 0; i < 20; i++)
    {
        err = vkllm_hashset_insert(set, i * 7); // Stride to increase collision probability
        ck_assert_int_eq(err, VKLLM_ERR_OK);
    }

    // Verify all keys
    for (uint64_t i = 0; i < 20; i++)
    {
        ck_assert(vkllm_hashset_contains(set, i * 7));
    }

    vkllm_hashset_free(set);
}
END_TEST

Suite *vkllm_test_hashset_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("HashSet");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_hashset_create_and_free);
    tcase_add_test(tc_core, test_hashset_insert_and_contains);
    tcase_add_test(tc_core, test_hashset_insert_duplicate);
    tcase_add_test(tc_core, test_hashset_remove);
    tcase_add_test(tc_core, test_hashset_clear);
    tcase_add_test(tc_core, test_hashset_resize);
    tcase_add_test(tc_core, test_hashset_large_keys);
    tcase_add_test(tc_core, test_hashset_collision_handling);

    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = vkllm_test_hashset_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
