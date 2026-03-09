#include "check.h"
#include "src/core/vkllm_hashmap.h"
#include <stdio.h>

START_TEST(test_hashmap_create_and_free)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_ptr_nonnull(map);
    ck_assert_int_eq(vkllm_hashmap_size(map), 0);
    ck_assert(vkllm_hashmap_empty(map));
    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_insert_and_get)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    err = vkllm_hashmap_insert(map, "hello", 42);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    err = vkllm_hashmap_insert(map, "world", 123);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    err = vkllm_hashmap_insert(map, "test", 999);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    ck_assert_int_eq(vkllm_hashmap_size(map), 3);
    ck_assert(!vkllm_hashmap_empty(map));

    uint64_t value;
    ck_assert(vkllm_hashmap_get(map, "hello", &value));
    ck_assert_int_eq(value, 42);
    ck_assert(vkllm_hashmap_get(map, "world", &value));
    ck_assert_int_eq(value, 123);
    ck_assert(vkllm_hashmap_get(map, "test", &value));
    ck_assert_int_eq(value, 999);

    ck_assert(!vkllm_hashmap_get(map, "nonexistent", &value));
    ck_assert(!vkllm_hashmap_contains(map, "nonexistent"));

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_insert_duplicate)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    err = vkllm_hashmap_insert(map, "key", 100);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashmap_size(map), 1);

    uint64_t value;
    ck_assert(vkllm_hashmap_get(map, "key", &value));
    ck_assert_int_eq(value, 100);

    err = vkllm_hashmap_insert(map, "key", 200);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashmap_size(map), 1);

    ck_assert(vkllm_hashmap_get(map, "key", &value));
    ck_assert_int_eq(value, 200);

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_remove)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    vkllm_hashmap_insert(map, "a", 10);
    vkllm_hashmap_insert(map, "b", 20);
    vkllm_hashmap_insert(map, "c", 30);
    ck_assert_int_eq(vkllm_hashmap_size(map), 3);

    err = vkllm_hashmap_remove(map, "b");
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashmap_size(map), 2);
    ck_assert(!vkllm_hashmap_contains(map, "b"));
    ck_assert(vkllm_hashmap_contains(map, "a"));
    ck_assert(vkllm_hashmap_contains(map, "c"));

    err = vkllm_hashmap_remove(map, "nonexistent");
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert_int_eq(vkllm_hashmap_size(map), 2);

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_clear)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    vkllm_hashmap_insert(map, "one", 1);
    vkllm_hashmap_insert(map, "two", 2);
    vkllm_hashmap_insert(map, "three", 3);
    ck_assert_int_eq(vkllm_hashmap_size(map), 3);

    vkllm_hashmap_clear(map);
    ck_assert_int_eq(vkllm_hashmap_size(map), 0);
    ck_assert(vkllm_hashmap_empty(map));
    ck_assert(!vkllm_hashmap_contains(map, "one"));
    ck_assert(!vkllm_hashmap_contains(map, "two"));
    ck_assert(!vkllm_hashmap_contains(map, "three"));

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_resize)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 4);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    char key[32];
    for (uint64_t i = 0; i < 100; i++)
    {
        snprintf(key, sizeof(key), "key_%lu", i);
        err = vkllm_hashmap_insert(map, key, i);
        ck_assert_int_eq(err, VKLLM_ERR_OK);
    }

    ck_assert_int_eq(vkllm_hashmap_size(map), 100);

    uint64_t value;
    for (uint64_t i = 0; i < 100; i++)
    {
        snprintf(key, sizeof(key), "key_%lu", i);
        ck_assert_msg(vkllm_hashmap_get(map, key, &value), "Key %s should exist", key);
        ck_assert_int_eq(value, i);
    }

    ck_assert(!vkllm_hashmap_contains(map, "key_200"));
    ck_assert(!vkllm_hashmap_contains(map, "nonexistent"));

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_long_keys)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    const char *long_keys[] = {
        "this_is_a_very_long_key_name_for_testing_purposes_1", "this_is_a_very_long_key_name_for_testing_purposes_2",
        "another_long_key_that_should_work_correctly_in_the_hash_map",
        "yet_another_key_with_some_special_chars_!@#$%^&*()", "unicode_key_\xe4\xb8\xad\xe6\x96\x87_test"};

    for (size_t i = 0; i < sizeof(long_keys) / sizeof(long_keys[0]); i++)
    {
        err = vkllm_hashmap_insert(map, long_keys[i], (uint64_t)i);
        ck_assert_int_eq(err, VKLLM_ERR_OK);
    }

    uint64_t value;
    for (size_t i = 0; i < sizeof(long_keys) / sizeof(long_keys[0]); i++)
    {
        ck_assert(vkllm_hashmap_get(map, long_keys[i], &value));
        ck_assert_int_eq(value, (uint64_t)i);
    }

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_collision_handling)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 8);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    char key[32];
    for (uint64_t i = 0; i < 20; i++)
    {
        snprintf(key, sizeof(key), "col_%lu", i * 7);
        err = vkllm_hashmap_insert(map, key, i * 7);
        ck_assert_int_eq(err, VKLLM_ERR_OK);
    }

    uint64_t value;
    for (uint64_t i = 0; i < 20; i++)
    {
        snprintf(key, sizeof(key), "col_%lu", i * 7);
        ck_assert(vkllm_hashmap_get(map, key, &value));
        ck_assert_int_eq(value, i * 7);
    }

    vkllm_hashmap_free(map);
}
END_TEST

START_TEST(test_hashmap_empty_key)
{
    struct vkllm_hashmap *map = NULL;
    vkllm_err_t err = vkllm_hashmap_new(&map, 16);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    err = vkllm_hashmap_insert(map, "", 0);
    ck_assert_int_eq(err, VKLLM_ERR_OK);

    uint64_t value;
    ck_assert(vkllm_hashmap_get(map, "", &value));
    ck_assert_int_eq(value, 0);

    err = vkllm_hashmap_insert(map, "", 42);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert(vkllm_hashmap_get(map, "", &value));
    ck_assert_int_eq(value, 42);

    err = vkllm_hashmap_remove(map, "");
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    ck_assert(!vkllm_hashmap_contains(map, ""));

    vkllm_hashmap_free(map);
}
END_TEST

Suite *vkllm_test_hashmap_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("HashMap");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_hashmap_create_and_free);
    tcase_add_test(tc_core, test_hashmap_insert_and_get);
    tcase_add_test(tc_core, test_hashmap_insert_duplicate);
    tcase_add_test(tc_core, test_hashmap_remove);
    tcase_add_test(tc_core, test_hashmap_clear);
    tcase_add_test(tc_core, test_hashmap_resize);
    tcase_add_test(tc_core, test_hashmap_long_keys);
    tcase_add_test(tc_core, test_hashmap_collision_handling);
    tcase_add_test(tc_core, test_hashmap_empty_key);

    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = vkllm_test_hashmap_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
