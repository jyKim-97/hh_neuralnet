INCDIR="../include/"

mpicc -o mpifor.o -c mpifor.c 
mpicc -Wall -g -O3 -I../include/ -I${MKLROOT}/include \
    -o main.out main.c \
    ${INCDIR}/model.o ${INCDIR}/utils.o ${INCDIR}/build.o \
    ${INCDIR}/rng.o ${INCDIR}/mt64.o ${INCDIR}/storage.o ${INCDIR}/measurement.o \
    ./mpifor.o -lm
#    -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
