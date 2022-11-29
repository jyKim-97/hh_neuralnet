#include "mpifor.h"

#define SIG_DONE 1
#define SIG_SEND 2
const int sig_end = -1;

// TODO: core 사용량 제한하는 코드 추가

int world_rank, world_size;


void init_mpi(int *argc, char ***argv){
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (1 == world_size){
        printf("Too small world size\n");
    }
}


void for_mpi(int nitr, void (*f) (int, void*), void *arg){
    if (0 == world_rank){
        control_tower(nitr);
    } else {
        iterate(f, arg);
    }
}


void control_tower(int nitr){
    int id_itr=0, id_source=0, buf;
    int num_end=0;
    int num_slave=world_size-1;

    MPI_Status status;
    while (num_end < num_slave){
        if (id_itr < num_slave){
            id_source = id_itr+1;
        } else {
            MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, SIG_DONE, MPI_COMM_WORLD, &status);
            id_source = status.MPI_SOURCE;
        }

        if (id_itr < nitr){
            MPI_Send(&id_itr, 1, MPI_INT, id_source, SIG_SEND, MPI_COMM_WORLD);
            id_itr++;
        } else {
            MPI_Send(&sig_end, 1, MPI_INT, id_source, SIG_SEND, MPI_COMM_WORLD);
            num_end++;
        }
    }
}


void iterate(void (*f) (int, void*), void *arg){
    int id_itr=0;
    MPI_Status status;

    MPI_Recv(&id_itr, 1, MPI_INT, 0, SIG_SEND, MPI_COMM_WORLD, &status);
    while (id_itr != -1){
        // function to iterate
        f(id_itr, arg);

        MPI_Send(&id_itr, 1, MPI_INT, 0, SIG_DONE, MPI_COMM_WORLD);
        MPI_Recv(&id_itr, 1, MPI_INT, 0, SIG_SEND, MPI_COMM_WORLD, &status);
    }
}


void end_mpi(){
    MPI_Finalize();
}