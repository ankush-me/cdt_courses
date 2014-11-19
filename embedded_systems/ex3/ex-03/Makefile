EXE = ex-03
SRC = $(wildcard *.c)
OBJ = $(addprefix build/,$(SRC:.c=.o))

all: $(EXE)

$(EXE): $(OBJ)
	@echo 'Building target: $(EXE)'
	@echo 'Invoking: C Linker'
	gcc -o $(EXE) $(OBJ)
	@echo 'Finished building target: $(EXE)'
	@echo ' '

build/%.o: %.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g0 -Wall -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

clean:
	rm -f build/*.o $(EXE)*
