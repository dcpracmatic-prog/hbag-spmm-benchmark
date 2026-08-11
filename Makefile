CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -fPIC
LDFLAGS ?= -lm

.PHONY: all clean

all: libhbag.so bench

libhbag.so: src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) -shared src/spmm.c -o libhbag.so $(LDFLAGS)

bench: src/bench.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) src/bench.c src/spmm.c -o bench $(LDFLAGS)

clean:
	rm -f bench libhbag.so *.o out.csr out.hbag
