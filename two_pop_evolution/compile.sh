make -C ../include/ main
# gcc -std=c11 -O2 -I../include -fPIC -shared -o "./simulation.so" simulation.c -L../lib -lhhnet -lm
mpicc -std=c11 -O2 -I../include -o run_mpi.out run_mpi.c ../include/mpifor.o -L../lib -lhhnet -lm
# gcc -I../include -Wall -g -o main.out simulation.c -L../lib/ -lhhnet -lm
