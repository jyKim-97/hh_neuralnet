cd ../include
make main
cd ../example_two_cellpop

gcc -O3 -g -Wall -c main.c -o main.o -I../include
gcc -Wall -g -O3 -o main.out main.o ../include/model.o ../include/utils.o ../include/build.o ../include/mt64.o ../include/storage.o