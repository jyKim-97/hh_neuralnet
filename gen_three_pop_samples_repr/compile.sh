make -C ../include main
mpicc -I ../include/ main.c ../include/mpifor.o -o main.out -L ../lib -lhhnet -lm