make -C ../include main
mpicc -I ../include/ main_template.c ../include/mpifor.o -o main_template.out -L ../lib -lhhnet -lm
mpicc -I ../include/ main.c ../include/mpifor.o -o main.out -L ../lib -lhhnet -lm