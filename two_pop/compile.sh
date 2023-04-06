make -C ../include/ main
gcc -Wall -O3 -I../include -o main.out main.c -L../lib -lhhnet -lm
