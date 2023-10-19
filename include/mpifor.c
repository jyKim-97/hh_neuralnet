#include "mpifor.h"

#define SIG_DONE 1
#define SIG_SEND 2
#define TIMER struct timeval

int sig_end = -1;

// TODO: core 사용량 제한하는 코드 추가

static double get_elapsed_time(TIMER _tic);
static void _print_job_start(int node_id, int job_id, int max_job);
static void _print_job_end(int node_id, int job_id, int max_job, TIMER _tic);

static TIMER *tics, tic_sub; // timer in total population set

int world_rank, world_size;


void init_mpi(int *argc, char ***argv){
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}


void for_mpi(int nitr, void (*f) (int, void*), void *arg){
    if (world_size == 1){
        for (int n=0; n<nitr; n++){
            gettimeofday(&tic_sub, NULL);
            _print_job_start(0, n, nitr);
            f(n, arg);
            _print_job_end(0, n, nitr, tic_sub);
        }
    } else {
        if (world_rank == 0){
            control_tower(nitr);
        } else {
            iterate(f, arg);
        }   
    }

    mpi_barrier();
}


void mpi_barrier(void){
    MPI_Barrier(MPI_COMM_WORLD);
}


void control_tower(int nitr){
    int id_itr=0, id_source=0, id_done;
    int num_end=0;
    int num_slave=world_size-1;

    tics = (TIMER*) malloc(sizeof(TIMER) * world_size);

    MPI_Status status;
    while (num_end < num_slave){
        if (id_itr < num_slave){
            id_source = id_itr+1;
        } else {
            MPI_Recv(&id_done, 1, MPI_INT, MPI_ANY_SOURCE, SIG_DONE, MPI_COMM_WORLD, &status);
            id_source = status.MPI_SOURCE;

            _print_job_end(id_source, id_itr, nitr, tics[id_source]);
        }

        if (id_itr < nitr){
            _print_job_start(id_source, id_itr, nitr);
            gettimeofday(&(tics[id_source]), NULL); // press timer
            MPI_Send(&id_itr, 1, MPI_INT, id_source, SIG_SEND, MPI_COMM_WORLD);
            id_itr++;
        } else {
            MPI_Send(&sig_end, 1, MPI_INT, id_source, SIG_SEND, MPI_COMM_WORLD);
            num_end++;
        }
    }

    free(tics);
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


static void _print_job_start(int node_id, int job_id, int max_job){
    time_t t_tmp = time(NULL);
    struct tm t = *localtime(&t_tmp);
    printf("[%d-%02d-%02d %02d:%02d:%02d] ", t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    printf("Node%3d Start job%5d/%d in pid %d\n", node_id, job_id, max_job, getpid());
}


static void _print_job_end(int node_id, int job_id, int max_job, TIMER _tic){
    double elapsed = get_elapsed_time(_tic);
    time_t t_tmp = time(NULL);
    struct tm t = *localtime(&t_tmp);
    printf("[%d-%02d-%02d %02d:%02d:%02d] ", t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    printf("Node%3d Done  job%5d/%d, elapsed=%.1fs\n", node_id, job_id, max_job, elapsed);
}


void print_job_start(int job_id, int max_job){
    _print_job_start(world_rank, job_id, max_job);
    gettimeofday(&tic_sub, NULL);
}


void print_job_end(int job_id, int max_job){
    _print_job_end(world_rank, job_id, max_job, tic_sub);
}


static double get_elapsed_time(TIMER _tic){
    int sec, msec;

    TIMER _toc;
    gettimeofday(&_toc, NULL);

    sec = _toc.tv_sec - _tic.tv_sec;
    msec = (_toc.tv_usec - _tic.tv_usec)/1e3;

    if (msec < 0){
        sec -= 1;
        msec += 1e3;
    }

    double dt = sec + ((double) msec) * 1e-3;

    return dt;
}