make -C ../include/ main
make -C ../include mpifor

# mpicc -Wall -O2 -I../include -o main.out main.c ../include/mpifor.o -L../lib -lhhnet -lm
mpicc -Wall -O2 -I../include -o main2d.out main2d.c ../include/mpifor.o -L../lib -lhhnet -lm