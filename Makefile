## `Makefile`

```makefile
CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall
LDFLAGS ?= -lm

.PHONY: all bench cache_probe run clean

all: bench cache_probe

bench: src/bench.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) src/bench.c src/spmm.c -o bench $(LDFLAGS)

cache_probe: tools/cache_probe.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) tools/cache_probe.c src/spmm.c -o cache_probe $(LDFLAGS)

run: bench
	./bench

clean:
	rm -f bench cache_probe *.o out.csr out.hbag
