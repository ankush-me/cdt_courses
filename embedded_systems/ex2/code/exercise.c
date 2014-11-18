#include <stdio.h>

void run_read_write_exercise(void);
void run_malloc_free_exercise(void);
void run_illegal_memory_exercise(void);
void run_function_pointers_exercise(void);
void run_callbacks_exercise(void);
void run_data_structures_exercise(void);

int main(void) {
  run_read_write_exercise();
  run_malloc_free_exercise();
  // Warning: This method will likely crash your application. Fix all errors and remove memory leaks.
  run_illegal_memory_exercise();
  run_function_pointers_exercise();
  run_callbacks_exercise();
  run_data_structures_exercise();

  return 0;
}
