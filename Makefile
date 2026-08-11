CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall -fPIC
LDFLAGS ?= -lm

.PHONY: all bench cache_probe shared run clean

all: bench cache_probe shared

bench: src/bench.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) src/bench.c src/spmm.c -o bench $(LDFLAGS)

cache_probe: tools/cache_probe.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) tools/cache_probe.c src/spmm.c -o cache_probe $(LDFLAGS)

shared: src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) -shared src/spmm.c -o libhbag.so $(LDFLAGS)

run: bench
	./bench

clean:
	rm -f bench cache_probe libhbag.so *.o out.csr out.hbag
