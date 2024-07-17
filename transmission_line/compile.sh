make -C ../include/ main
make -C ../include mpifor

# gcc -Wall -g -O2 -I../include -o main.out main.c -L../lib -lhhnet -lm
mpicc -Wall -g -O2 -I../include -o main.out main.c ../include/mpifor.o -L../lib -lhhnet -lm