#include <stdio.h>
#include <stdlib.h>

// #define NDIM 5

/*
$ gcc -c -fPIC -Wall estimate_prob.c
$ gcc -shared -o estimate_prob.so estimate_prob.o 
*/


// int convert_id(int size[NDIM], int id[NDIM]);
int convert_id(int ndim, int size[], int id[]);
int estimate_full_hist(int N, int *x, int *y, int *class_id, int nbin, int nlag_max, int *ncount);
int estimate_full_hist_2d(int N, int *x, int *y, int *class_id, int nbin, int nlag_max, int *ncount);

// int convert_id(int size[NDIM], int id[NDIM]){
//     // int id_flat = id[0];
//     int id_flat = 0;
//     int nstack = 1;
//     for (int n=NDIM-1; n>-1; n--){
//         if ((id[n] < 0) || (id[n] > size[n])) return -1;
        
//         id_flat += id[n] * nstack;
//         nstack *= size[n];
//     }

//     return id_flat;
// }


int convert_id(int ndim, int size[], int id[]){
    // int id_flat = id[0];
    int id_flat = 0;
    int nstack = 1;
    for (int n=ndim-1; n>-1; n--){
        if ((id[n] < 0) || (id[n] > size[n])) return -1;
        
        id_flat += id[n] * nstack;
        nstack *= size[n];
    }

    return id_flat;
}



int estimate_full_hist(int N, int *x, int *y, int *class_id, int nbin, int nlag_max, int *ncount){
    /*
    
    args: 
        N: number of dataset
        

    returns:
        ncount: (nbin, nbin, nbin, nlag_max, 2)

    */
    const int ndim = 5;
    int size[] = {nbin, nbin, nbin, nlag_max, 2};

    int *nx_prv = x;
    int *ny_prv = y;
    int *cc_prv = class_id;

    for (int n=0; n<N-nlag_max; n++){
        int *nx_cur = nx_prv+1;
        int *ny_cur = ny_prv+1;
        int *cc_cur = cc_prv+1;

        for (int nl=0; nl<nlag_max; nl++){
            if (*cc_prv != *cc_cur) break;
            
            int id0[] = {*ny_cur, *ny_prv, *nx_prv, nl, 0};
            int id0_f = convert_id(ndim, size, id0);
            if (id0_f != -1) ncount[id0_f]++;

            // printf("#%03d, (%d,%d,%d,%d,%d) -> %d\n", n, *ny_cur, *ny_prv, *nx_prv, nl, 0, id0_f);

            int id1[] = {*nx_cur, *nx_prv, *ny_prv, nl, 1};
            int id1_f = convert_id(ndim, size, id1);
            if (id1_f != -1) ncount[id1_f]++;

            nx_cur++; ny_cur++; cc_cur++;
        }

        nx_prv++; ny_prv++; cc_prv++;
    }

    return 0;
}


int estimate_full_hist_2d(int N, int *x, int *y, int *class_id, int nbin, int nlag_max, int *ncount){
    /*
    Compute TE-2D
    TE[...,0] = I(Y(t), X(t-t1); Y(t-t2))
    TE[...,1] = I(X(t), Y(t-t1); X(t-t2))
    args: 
        N: number of dataset

    returns:
        ncount: (nbin, nbin, nbin, nlag_max, 2)

    */
    const int ndim = 6;
    int size[] = {nbin, nbin, nbin, nlag_max, nlag_max, 2};

    int *nx_prv = x;
    int *ny_prv = y;
    int *cc_prv = class_id;

    for (int n=0; n<N-nlag_max; n++){
        for (int ntp=0; ntp<2; ntp++){

            int *pt1, *pt2, *pt1_prv, *pt2_prv;
            int *pt_cc1, *pt_cc2;

            if (ntp == 0){
                pt1_prv = ny_prv;
                pt2_prv = nx_prv;
                
            } else{
                pt1_prv = nx_prv;
                pt2_prv = ny_prv;
            }
            
            pt1 = pt1_prv+1;
            pt_cc1 = cc_prv+1;
            for (int n1=0; n1<nlag_max; n1++){
                if (*pt_cc1 != *cc_prv) break;
                
                pt2 = pt2_prv+1;
                pt_cc2 = cc_prv+1;
                for (int n2=0; n2<nlag_max; n2++){
                    if (*pt_cc2 != *cc_prv) break;

                    int id[] = {*pt1, *pt2, *pt1_prv, n1, n2, ntp};
                    int id_f = convert_id(ndim, size, id);
                    // printf("id_f: %d\n", id_f);
                    if (id_f != -1) ncount[id_f]++;
                    
                    pt_cc2++; pt2++;
                }
                pt_cc1++; pt1++;
            }
        }

        nx_prv++; ny_prv++; cc_prv++;
    }

    return 0;
}


int main(){

    int N = 100;
    int *a1 = (int*) calloc(N, sizeof(int));
    int *a2 = (int*) calloc(N, sizeof(int));
    int *cc = (int*) calloc(N, sizeof(int));

    int nbin = 10;
    int nlag_max = 5;
    for (int n=0; n<N; n++){
        a1[n] = n % nbin;
        a2[n] = (n+2) % nbin;
    }

    // int nlen = nbin*nbin*nbin*nlag_max*2;
    // int *out = (int*) calloc(nlen, sizeof(int));
    // estimate_full_hist(N, a1, a2, cc, nbin, nlag_max, out);

    int nlen = nbin*nbin*nbin*nlag_max*nlag_max*2;
    int *out = (int*) calloc(nlen, sizeof(int));
    estimate_full_hist_2d(N, a1, a2, cc, nbin, nlag_max, out);

    printf("out0: %d\n", out[10]);

    FILE *fp = fopen("./out.dat", "wb");
    fwrite(out, sizeof(int), nlen, fp);
    fclose(fp);
}
