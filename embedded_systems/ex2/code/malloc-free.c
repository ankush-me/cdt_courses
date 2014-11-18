#include <stdio.h>
#define N 100

int *reserve_memory () {
	return (int *) malloc(N * sizeof ( int ));
}

void initialise_memory(int *data) {
	int i;
	for(i = 0; i < 100; ++i) data[i] = 0;
 }

void manipulate_data(int *data) {
	data [0] = 1;
	data [1] = 1;
	int i;
	for(i = 2; i < 100; ++i) {
		data[i] = data[i-2] + data[i-1];
	}
}

void release_memory(int *data) {
	free(data);
}


// TODO: Write fibonacci code as illustrated on exercise sheet.
void print_fib() { 
	int *d = reserve_memory();
	initialise_memory(d);
	manipulate_data(d);
	int i;
	for(i=0; i<N; i++) {
		printf("%u ", d[i]);
	}
	printf("\n");
	release_memory(d);
}

void run_malloc_free_exercise(void) {
  puts("Malloc/free exercises");
  puts("------------------------------\n");
  // TODO: Print all generated Fibonacci values
  print_fib();
  puts("\n\n");
}
