#include "mini-assert/mini-assert.h"

#define DATA_SIZE 10
static int writeIndex = 0;
static int readIndex = 0;
static char data[DATA_SIZE];

void write(const char *buffer, unsigned int size) {
    // TODO: Write buffer content to data. Implement a circular buffer,
    // overwriting data at the beginning if you reach the end.
  for(int i=0; i<size; i++) {
    data[writeIndex] = buffer[i];
    writeIndex = (writeIndex+1)%DATA_SIZE;
  }
}

void read(char *buffer, unsigned int size) {
  // TODO: Read from data at current read index.
  // Also follow a circular buffer mentality.
  for(int i=0; i<size; i++) {
    buffer[i] = data[readIndex];
    readIndex = (readIndex+1)%DATA_SIZE;
  }
}

void run_read_write_exercise(void) {
  puts("Read/Write exercises");
  puts("------------------------------\n");

  write("Hello World!", 10);
  char buffer[8];
  read(buffer, sizeof(buffer));
  assert_equal_buffer("Hello Wo", buffer, sizeof(buffer));
  write("Goodbye", 7);
  read(buffer, sizeof(buffer));
  assert_equal_buffer("rlGoodby", buffer, sizeof(buffer));
  puts("\n");
}
