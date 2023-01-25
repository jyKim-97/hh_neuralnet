make -C ../include/ main
make -C ../include mpifor
mpicc -Wall -O2 -I../include -o main.out main.c -L../lib -lhhnet -lm
