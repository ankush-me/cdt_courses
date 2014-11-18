EXE = ex-01
SRC = $(wildcard *.c)
OBJ = $(addprefix build/,$(SRC:.c=.o))

all:
	gcc -o $(EXE) exercise.c

clean:
	rm -f $(EXE)*
