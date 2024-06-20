CC = gcc
# CFLAGS = -D USE_MKL -std=c11 -g -Wall -O3 -I${MKLROOT}/include
CFLAGS = -std=c11 -g -Wall -O2
# LDFLAGS = -L${MKLROOT}/include -lmkl_intel_lp64 -lmkl_core -lmkl_squential -lm
LDFLAGS = -lm 

SRCS = mt64.c utils.c ntk.c storage.c measurement2.c rng.c model2.c neuralnet.c 

OBJS = $(SRCS:.c=.o)
TARGET = ../lib/libhhnet.a

main: $(OBJS)
# rm -f ../lib/libhhnet.a
	ar -rc $(TARGET) $(OBJS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) -o $@ -c $*$..c 
# $(LDFLAGS)

mpifor:
	mpicc -g -o mpifor.o -c mpifor.c

# rng.o: rng.c
# 	$(CC) $(CFLAGS) -o rng.o -c rng.c $(LDFLAGS) -I${MKLROOT}/include 
# -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_squential

clean:
	rm -f *.o *.out



# all: NNS	
# NNS: NNS.c NNS.h parameters.h ode.o statistics.o random.o 
# 	${CC} ${FLAGS} NNS.c random.o ode.o statistics.o -o NNS -lm

# algebra: algebra.c algebra.h
# 	${CC} ${FLAGS} -

# ode.o: ode.c ode.h parameters.h 
# 	${CC} ${FLAGS} -c ode.c -o ode.o
# random.o: random.c random.h
# 	${CC} ${FLAGS} -c random.c -o random.o
# statistics.o: statistics.c statistics.h
# 	${CC} ${FLAGS} -c statistics.c -o statistics.o
# clean:
# 	rm -f core *~ *.o  gmon.out *.tmp.* *.out *.dat
