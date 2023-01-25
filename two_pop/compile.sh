make -C ../include/ main
make -C ../include mpifor

rm -f ./tmp/*
mpicc -Wall -O2 -I../include -o main.out main.c -L../lib -lhhnet -lm
