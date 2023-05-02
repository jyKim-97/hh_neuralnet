if [[ "$1" == "multi" ]];
then
    make -C ../include/ mpifor
    CFLAGS="mpicc"
    obj="../include/mpifor.o"
    opt="-D MULTI"
else
    CFLAGS="gcc"
    obj=""
    opt=""
fi    

make -C ../include/ main

com="$CFLAGS -Wall -O2 $opt -I../include -o main.out main.c $obj -L../lib -lhhnet -lm"
echo $com
$com

