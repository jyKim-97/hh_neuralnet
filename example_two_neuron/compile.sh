gcc -O3 -c main.c -o main.o -I../include
gcc -Wall -O3 -o main.out main.o ../include/model.o ../include/utils.o ../include/build.o ../include/mt64.o