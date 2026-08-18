.PHONY: all bench cache_probe shared hbag/libhbag.so libhbag.so run run-all test test-c test-py install clean distclean help
CC ?= gcc
CFLAGS ?= -O3 -march=native -mavx2 -mfma -fopenmp -Wall -fPIC
LDFLAGS ?= -lm

all: bench cache_probe hbag/libhbag.so
	@echo "[OK] bench + cache_probe + hbag/libhbag.so"

bench: src/bench.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) src/bench.c src/spmm.c -o bench $(LDFLAGS)

cache_probe: tools/cache_probe.c src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) tools/cache_probe.c src/spmm.c -o cache_probe $(LDFLAGS)

shared: hbag/libhbag.so

hbag/libhbag.so: src/spmm.c src/spmm.h
	@mkdir -p hbag
	$(CC) $(CFLAGS) -shared src/spmm.c -o hbag/libhbag.so $(LDFLAGS)
	@nm -D --defined-only hbag/libhbag.so | awk '/T (spmm_hbag|spmm_csr_std|generate_sparse)/{print "    " $$3}'

libhbag.so: src/spmm.c src/spmm.h
	$(CC) $(CFLAGS) -shared src/spmm.c -o libhbag.so $(LDFLAGS)

run: bench
	./bench --csv /tmp/bench_results.csv

run-all: bench
	./bench --csv /tmp/bench_results_t1.csv --threads 1
	./bench --csv /tmp/bench_results_t2.csv --threads 2
	./bench --csv /tmp/bench_results_t4.csv --threads 4
	@echo "[OK] /tmp/bench_results_t{1,2,4}.csv"

test: test-c test-py

test-c: bench
	./bench --self-test

test-py:
	PYTHONPATH=. python3 -m pytest tests/ -v

install:
	pip install -e . --no-build-isolation

clean:
	rm -f bench cache_probe libhbag.so hbag/libhbag.so out.csr out.hbag
	rm -rf hbag/__pycache__ tests/__pycache__ .pytest_cache build *.egg-info

distclean: clean
	rm -f /tmp/bench_results*.csv /tmp/repro_*.csv /tmp/big.csv /tmp/single_short.csv /tmp/out.csr /tmp/out.hbag

help:
	@echo "Targets: all bench cache_probe shared hbag/libhbag.so libhbag.so run run-all test test-c test-py install clean distclean help"
