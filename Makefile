#
# Makefile for nncp
# Copyright (c) 2018-2019 Fabrice Bellard
#
#CONFIG_WIN32=y
#CONFIG_ASAN=y
#CONFIG_GPROF=y

ifdef CONFIG_WIN32
CROSS_PREFIX=x86_64-w64-mingw32-
EXE=.exe
LIBEXT=.lib
else
LIBEXT=.a
endif

CC=$(CROSS_PREFIX)gcc
AR=$(CROSS_PREFIX)ar
CFLAGS_VERSION:=-DCONFIG_VERSION=\"$(shell cat VERSION)\"
CFLAGS=-O3 -Wall -Wpointer-arith -g -fno-math-errno -fno-trapping-math -MMD -Wno-format-truncation $(CFLAGS_VERSION)
LDFLAGS=
PROGS=preprocess$(EXE)
LIBS+=-lm

ifdef CONFIG_ASAN
CFLAGS+=-fsanitize=address -fno-omit-frame-pointer
LDFLAGS+=-fsanitize=address -fno-omit-frame-pointer
endif

all: $(PROGS)

clean:
	rm -f *.o *.d $(PROGS)

preprocess$(EXE): preprocess.o
	$(CC) $(LDFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(wildcard *.d)
