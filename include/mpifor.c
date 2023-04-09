#include "mpifor.h"

#define SIG_DONE 1
#define SIG_SEND 2
int sig_end = -1;

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

static struct timeval tic, toc;
static double get_dt_mpi(void);

void print_job_start(int job_id, int max_job){
    time_t t_tmp = time(NULL);
    struct tm t = *localtime(&t_tmp);
    printf("[%d-%02d-%02d %02d:%02d:%02d] ", t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    printf("Node%3d Start job%5d/%d in pid %d\n", world_rank, job_id, max_job, getpid());
    gettimeofday(&tic, NULL);
}


void print_job_end(int job_id, int max_job){
    double elapsed = get_dt_mpi();

    time_t t_tmp = time(NULL);
    struct tm t = *localtime(&t_tmp);
    printf("[%d-%02d-%02d %02d:%02d:%02d] ", t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    printf("Node%3d Done  job%5d/%d, elapsed=%.1fs\n", world_rank, job_id, max_job, elapsed);
}


double get_dt_mpi(void){
    gettimeofday(&toc, NULL);

    int sec, msec;

    sec = toc.tv_sec - tic.tv_sec;
    msec = (toc.tv_usec - tic.tv_usec)/1e3;

    if (msec < 0){
        sec -= 1;
        msec += 1e3;
    }

    double dt = sec + ((double) msec) * 1e-3;

    return dt;
}