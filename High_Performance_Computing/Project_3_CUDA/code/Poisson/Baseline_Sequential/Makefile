# Makefile
#
TARGET_J  = poisson_j		# Jacobi


SOURCES	= main.c print.c alloc3d.c 
OBJECTS	= print.o alloc3d.o 
MAIN_J	= main_j.o

OBJS_J	= $(MAIN_J) jacobi.o


# options and settings for the GCC compilers
#
CC	= cc -std=c99
DEFS	= 
OPT	= -Wall -g -O3 -fopenmp
IPO	= 
ISA	= 
CHIP	= 
ARCH	= 
PARA	= 
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) $(XOPTS)
LDFLAGS = -lm 

all: $(TARGET_J) 

$(TARGET_J): $(OBJECTS) $(OBJS_J)
	$(CC) -o $@ $(CFLAGS) $(OBJS_J) $(OBJECTS) $(LDFLAGS)


$(MAIN_J):
	$(CC) -o $@ -D_JACOBI $(CFLAGS) -c main.c 


clean:
	@/bin/rm -f core *.o *~

realclean: clean
	@/bin/rm -f $(TARGET_J)  $(TARGET_GS)

# DO NOT DELETE

main_j.o: main.c print.h jacobi.h 
print.o: print.h
