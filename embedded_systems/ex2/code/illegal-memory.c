#include <stdlib.h>
#include "malloc.h"
#include <stdio.h>
#include <string.h>

void error1(void) {
  puts("Q1: Find the error in this code segment. It should print the range 0-9 twice.\n");

  int *i = malloc(sizeof(int));
  int *j = malloc(sizeof(int));

  int **p = &i;
  for (**p = 0; **p < 10; ++**p) {
    printf("%d ", **p);
  }
  puts("\n");
  free(i);

  p = &j;
  for (**p = 0; **p < 10; ++**p) {
    printf("%d ", **p);
  }
  free(*p);
  free(p);
  puts("\n");
}

void output(char* str) {
  if (str) {
   str[0] = 'J';
  puts(str); 
  }
}

void error2(void) {
  puts("Q2: Find the error in this code segment. It should print \"Jello World\" once and \"Joodbye!\" twice.\n");

  char data[] = "Hello World!";
  output(data);

  char data2[] = "Goodbye!";
  output(data2);

  // const pointer to mutable data:
  char * const const_data = data2;
  output(const_data);
  // The above works. However, the following:
  // char * const const_data = "Goodbye!"
  // won't work because the string literal is stored in
  // read-only memory. And what we really have is a reference to it.
  // This is due to the fact that we are using -o3 in compilation.
}

void error3(void) {
  char *data= 0;
  int sizes[] = { 1, 32, 512, 1024, 1048576 };
  data = malloc(sizes[0]);
  memset(data, rand(), sizes[0]);
  int i;
  for (i = 1; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    //free(data);
    data = realloc(data, sizes[i]);
    memset(data, rand(), sizes[i]);
  }
  free(data);
}

void run_illegal_memory_exercise(void) {
  puts("Illegal memory exercises");
  puts("------------------------------\n");
  error1();
  error2();
  error3();
  puts("\n");
}
