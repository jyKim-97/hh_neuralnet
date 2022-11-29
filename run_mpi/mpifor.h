#include <stdio.h>
#include "mpi.h"

extern int world_rank, world_size;

void init_mpi(int *argc, char ***argv);
void for_mpi(int nitr, void (*f) (int, void*), void *arg);
void control_tower(int nitr);
void iterate(void (*f) (int, void*), void *arg);
void end_mpi();