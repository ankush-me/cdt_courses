#include <stdio.h>

void output1(const char *str) {
  puts(str);
}

int output2(unsigned int value, const char *str) {
  return printf("Value: %d, Str: %s\n", value, str);
}

// TODO: Replace "typeof(output1) *o1" and "typeof(output2) *o2" with explicitly typed parameters.
void execute_output(typeof(output1) *o1, typeof(output2) *o2) {
  o1("Hello World!");
  o2(999, "Goodbye!");
}

// explicit typed parameters:
void execute_output2(void (*o1)(char*) , int (*o2)(unsigned int, const char*) ) {
  o1("Hello World!");
  o2(999, "Goodbye!");
}

void run_function_pointers_exercise(void) {
  puts("Function pointers exercises");
  puts("------------------------------\n");
  execute_output(output1, output2);
  execute_output2(output1, output2);
  puts("\n\n");
}
