#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define assert_equal_delta(expected, actual) printf("%4.1f == %4.1f: %s\n", expected, actual, abs(expected - actual) < 0.0078125 ? "OK" : "FAILURE")
#define assert_equal_bool(expected, actual) printf("   %s ==    %s: %s\n", expected ? "t" : "f", actual ? "t" : "f", (expected != 0) == (actual != 0) ? "OK" : "FAILURE")
#define assert_equal_hex(expected, actual) printf("0x%08x == 0x%08x: %s\n", expected, actual, expected == actual ? "OK" : "FAILURE")
#define assert_equal_integer(expected, actual) printf("% 4d == % 4d: %s\n", expected, actual, expected == actual ? "OK" : "FAILURE")
#define assert_equal_buffer(expected, actual, size) printf("expected == actual: %s\n", strncmp(expected, actual, size) == 0 ? "OK" : "FAILURE")
