arg=$1
out="main$arg.out"
# echo $args # should be _normal _mfast, _mslow

make -C ../include main
# mpicc  -Wall -o3 -I -D\"TEST\" ../include/ main.c ../include/mpifor.o -o main_tmp.out -L ../lib -lhhnet -lm
mpicc  -Wall -o3 -DPTYPE=\"$arg\" -I ../include/ main.c ../include/mpifor.o -o $out -L ../lib -lhhnet -lm

echo "Compiled into $out"


# Run simulation by
# mpirun -np [NUM] --hostfile hostfile ./main$arg.out --fdir [TARGET DIR]