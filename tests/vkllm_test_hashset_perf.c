#include "check.h"
#include "src/core/vkllm_hashset.h"
#include "src/core/vkllm_array.h"
#include <stdio.h>
#include <time.h>

// Performance comparison test: hash set vs array linear search
START_TEST(test_hashset_performance_comparison)
{
    const size_t num_elements = 1000;
    const size_t num_lookups = 10000;
    
    // Setup hash set
    struct vkllm_hashset *set = NULL;
    vkllm_err_t err = vkllm_hashset_new(&set, num_elements);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    
    // Setup array
    struct vkllm_array_ptr *array = NULL;
    err = vkllm_array_ptr_new(&array, num_elements);
    ck_assert_int_eq(err, VKLLM_ERR_OK);
    
    // Populate both with the same data
    for (size_t i = 0; i < num_elements; i++) {
        uint64_t key = i * 7; // Some pattern
        vkllm_hashset_insert(set, key);
        vkllm_array_ptr_append(array, (void *)(uintptr_t)key);
    }
    
    printf("\n=== Performance Comparison ===\n");
    printf("Elements: %zu, Lookups: %zu\n\n", num_elements, num_lookups);
    
    // Test hash set lookups
    clock_t start = clock();
    size_t found_count = 0;
    for (size_t i = 0; i < num_lookups; i++) {
        uint64_t key = (i % num_elements) * 7;
        if (vkllm_hashset_contains(set, key)) {
            found_count++;
        }
    }
    clock_t end = clock();
    double hashset_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    printf("Hash Set:\n");
    printf("  Time: %.3f ms\n", hashset_time);
    printf("  Found: %zu/%zu\n\n", found_count, num_lookups);
    
    // Test array linear search
    start = clock();
    found_count = 0;
    for (size_t i = 0; i < num_lookups; i++) {
        uint64_t key = (i % num_elements) * 7;
        for (size_t j = 0; j < array->used_n; j++) {
            if ((uint64_t)(uintptr_t)array->data[j] == key) {
                found_count++;
                break;
            }
        }
    }
    end = clock();
    double array_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    printf("Array (Linear Search):\n");
    printf("  Time: %.3f ms\n", array_time);
    printf("  Found: %zu/%zu\n\n", found_count, num_lookups);
    
    double speedup = array_time / hashset_time;
    printf("Speedup: %.2fx faster with hash set\n", speedup);
    printf("==============================\n\n");
    
    // Hash set should be significantly faster
    ck_assert(hashset_time < array_time);
    
    vkllm_hashset_free(set);
    vkllm_array_ptr_free(array);
}
END_TEST

Suite *vkllm_test_hashset_perf_suite(void)
{
    Suite *s;
    TCase *tc_perf;

    s = suite_create("HashSet Performance");
    tc_perf = tcase_create("Performance");

    // Increase timeout for performance tests
    tcase_set_timeout(tc_perf, 30);
    
    tcase_add_test(tc_perf, test_hashset_performance_comparison);

    suite_add_tcase(s, tc_perf);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = vkllm_test_hashset_perf_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
