CC = cc

CCFLAGS = -Wall -O3

all: test-universal simple

universal.o: universal.h universal.c
	$(CC) $(CCFLAGS) -c -o universal.o universal.c

test-universal: test-universal.c universal.o
	$(CC) $(CCFLAGS) -o test-universal test-universal.c universal.o -lm

simple: simple.c universal.o
	$(CC) $(CCFLAGS) -o simple simple.c universal.o -lm	

clean:
	rm test-universal simple universal.o 


