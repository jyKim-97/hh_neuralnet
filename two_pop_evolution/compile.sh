make -C ../include/ main
gcc -O2 -I../include -fPIC -shared -o "./simulation.so" simulation.c -L../lib -lhhnet -lm
# gcc -I../include -Wall -g -o main.out simulation.c -L../lib/ -lhhnet -lm