make -C ../include/ main

if [[ "$1" == "test" ]]; then
    echo "compile for testing"
    gcc -DTEST -Wall -g -O2 -I../include -o main.out main.c -L../lib -lhhnet -lm
else
    echo "compile for normal run"
    make -C ../include mpifor
    mpicc -Wall -g -O2 -I../include -o main.out main.c ../include/mpifor.o -L../lib -lhhnet -lm
fi
