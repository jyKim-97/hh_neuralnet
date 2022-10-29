#FLAGS=-Wall -Winline -O3
# FLAGS=-Wall -Winline -O3 -funroll-loops
#FLAGS=-g -Wall -pg
#FLAGS=-g
CC = gcc
CFLAGS = -g -s -Wall -O3
LDFLAGS = -lm

SRCS = model.c build.c mt64.c utils.c storage.c
OBJS = $(SRCS:.c=.o)
TARGET = ../lib/libhhnet.a

main: $(OBJS)
	ar -rc $(TARGET) $(OBJS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) -o $@ -c $*$..c $(LDFLAGS)




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
