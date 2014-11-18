#include <stdio.h>

// TODO: Define callback type for functions like "first" and "second" below.
typedef void * callbackt;
// TODO: Create storage for callbacks.
#define CALLBACK_STORAGE_SIZE 10
callbackt fs[CALLBACK_STORAGE_SIZE];
static int fi = 0;

void event_occurred(void) {
  // TODO: Implement callback execution.
  for (int i=0; i<fi; i++)
    if (fs[i]) {
      void (*f)(void) = fs[i];
      f();
     }
}

void add_callback(callbackt callback) {
  // TODO: Add callback to created storage.
  fs[fi++] = callback;
}

void remove_callback(callbackt callback) {
  // TODO: Remove callback from storage.
  for (int i=0; i<=fi; i++) {
    if (fs[i]==callback) 
      fs[i] = 0;
  }
}

void first(void) {
  puts("first");
}

void second(void) {
  puts("second");
}

void run_callbacks_exercise(void) {
  puts("Callback exercises");
  puts("------------------------------\n");
  add_callback(first);
  add_callback(second);
  event_occurred();
  remove_callback(second);
  event_occurred();
  puts("\n");
}
