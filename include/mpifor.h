#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int world_rank, world_size;

void init_mpi(int *argc, char ***argv);
void for_mpi(int nitr, void (*f) (int, void*), void *arg);
void control_tower(int nitr);
void iterate(void (*f) (int, void*), void *arg);
void end_mpi();

// logging
void print_job_start(int job_id, int max_job);
void print_job_end(int job_id, int max_job);

#ifdef __cplusplus
}
#endif