#include "parse.h"
#include "stdlib.h"
#include "stdio.h"
#include <string.h>
#include <algorithm>
#include <sys/time.h>
#include "util.h"

bool doCompare(CooTuple elem1, CooTuple elem2)
{
    if (elem1.row < elem2.row) {
        return true;
    }
    return false;
}

ell_array *csr2ell(csr_array *csr, int num_nodes, int num_edges, int fill)
{
    int size, maxheight = 0;
    for (int i = 0; i < num_nodes; i++) {
        size = csr->row_array[i + 1] - csr->row_array[i];
        if (size > maxheight)
            maxheight = size;
    }

    ell_array *ell = (ell_array *)malloc(sizeof(ell_array));
    if (!ell) printf("malloc failed");

    ell->max_height = maxheight;
    ell->num_nodes = num_nodes;

    ell->col_array = (int*)malloc(sizeof(int) * maxheight * num_nodes);
    ell->data_array = (int*)malloc(sizeof(int) * maxheight * num_nodes);


    for (int i = 0; i < maxheight * num_nodes; i++) {
        ell->col_array[i] = 0;
        ell->data_array[i] = fill;
    }

    for (int i = 0; i < num_nodes; i++) {
        int start = csr->row_array[i];
        int end = csr->row_array[i + 1];
        int lastcolid = 0;
        for (int j = start; j < end; j++) {
            int colid = csr->col_array[j];
            int data = csr->data_array[j];
            ell->col_array[i + (j - start) * num_nodes] = colid;
            ell->data_array[i + (j - start) * num_nodes] = data;
            lastcolid = colid;
        }
        for (int j = end; j < start + maxheight; j++) {
            ell->col_array[i + (j - start) * num_nodes] = lastcolid;
            ell->data_array[i + (j - start) * num_nodes] = fill;
        }
    }

    return ell;

}

csr_array *parseMetis(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{

    int cnt = 0;
    unsigned int lineno = 0;
    char *line = (char *)malloc(8192);
    int num_edges = 0, num_nodes = 0;
    int *col_cnt = NULL;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 8192, fptr)) {
        int head, tail, weight = 0;
        CooTuple temp;

        if (line[0] == '%') continue; // skip comment lines

        if (lineno == 0) { //the first line

            sscanf(line, "%d %d", p_num_nodes, p_num_edges);
            col_cnt = (int *)malloc(*p_num_nodes * sizeof(int));
            if (!col_cnt) {
                printf("memory allocation failed for col_cnt\n");
                exit(1);
            }
            memset(col_cnt, 0, *p_num_nodes * sizeof(int));

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }
            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;


            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
        } else if (lineno > 0) { //from the second line

            char *pch;
            pch = strtok(line , " ,.-");
            while (pch != NULL) {
                head = lineno;
                tail = atoi(pch);
                if (tail <= 0)  break;

                if (tail == head) printf("reporting self loop: %d, %d\n", lineno + 1, lineno);

                temp.row = head - 1;
                temp.col = tail - 1;
                temp.val = weight;

                col_cnt[head - 1]++;
                tuple_array[cnt++] = temp;

                pch = strtok(NULL, " ,.-");

            }
        }

#ifdef VERBOSE
        printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif

        lineno++;

    }

    //std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }
        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;

    }
    row_array[row_cnt] = idx;

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;
    csr->col_cnt = col_cnt;

    fclose(fptr);
    free(tuple_array);

    return csr;

}

csr_array *parseCOO(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
    int num_nodes = 0, num_edges = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            sscanf(line, "%c %s %d %d", &p, sp, p_num_nodes, p_num_edges);

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            break;

        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            CooTuple temp;
            temp.row = head - 1;
            temp.col = tail - 1;
            temp.val = weight;
            tuple_array[cnt++] = temp;
            if (!directed) {
                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
            break;
        default:
            fprintf(stderr, "exiting loop\n");
            break;
        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }

        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }

    row_array[row_cnt] = idx;

    fclose(fptr);
    free(tuple_array);

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    return csr;

}

double_edges *parseMetis_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[4096];
    int num_edges = 0, num_nodes = 0;
    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 4096, fptr)) {
        int head, tail, weight = 0;
        CooTuple temp;

        if (line[0] == '%') continue; // skip comment lines

        if (lineno == 0) { //the first line

            sscanf(line, "%d %d", p_num_nodes, p_num_edges);
            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            if (!tuple_array) printf("xxxxxxxx\n");

        } else if (lineno > 0) { //from the second line
            char *pch;
            pch = strtok(line , " ,.-");
            while (pch != NULL) {
                head = lineno;
                tail = atoi(pch);
                if (tail <= 0) break;

                if (tail == head) printf("reporting self loop: %d, %d\n", lineno + 1, lineno);

                temp.row = head - 1;
                temp.col = tail - 1;
                temp.val = weight;

                tuple_array[cnt++] = temp;

                pch = strtok(NULL, " ,.-");
            }
        }

#ifdef VERBOSE
        printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif

        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *edge_array1 = (int *)malloc(num_edges * sizeof(int));
    int *edge_array2 = (int *)malloc(num_edges * sizeof(int));

    for (int i = 0; i < num_edges; i++) {
        edge_array1[i] = tuple_array[i].row;
        edge_array2[i] = tuple_array[i].col;
    }

    fclose(fptr);
    free(tuple_array);

    double_edges *de = (double_edges *)malloc(sizeof(double_edges));
    de->edge_array1 = edge_array1;
    de->edge_array2 = edge_array2;

    return de;

}

double_edges *parseCOO_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
    int num_nodes = 0, num_edges = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            sscanf(line, "%c %s %d %d", &p, sp, p_num_nodes, p_num_edges);

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            break;
        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            CooTuple temp;
            temp.row = head - 1;
            temp.col = tail - 1;
            temp.val = weight;
            tuple_array[cnt++] = temp;
            if (!directed) {
                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
            break;
        default:
            fprintf(stderr, "exiting loop\n");
            break;

        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *edge_array1 = (int *)malloc(num_edges * sizeof(int));
    int *edge_array2 = (int *)malloc(num_edges * sizeof(int));

    for (int i = 0; i < num_edges; i++) {
        edge_array1[i] = tuple_array[i].row;
        edge_array2[i] = tuple_array[i].col;
    }

    fclose(fptr);
    free(tuple_array);

    double_edges *de = (double_edges *)malloc(sizeof(double_edges));
    de->edge_array1 = edge_array1;
    de->edge_array2 = edge_array2;

    return de;
}

csr_array *parseMM(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed, bool weight_flag)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128];
    int num_nodes = 0, num_edges = 0, num_nodes2 = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        if (line[0] == '%') continue;
        if (lineno == 0) {
            sscanf(line, "%d %d %d", p_num_nodes, &num_nodes2, p_num_edges);
            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            if (!tuple_array) {
                printf("tuple array not allocated succesfully\n");
                exit(1);
            }

        }
        if (lineno > 0) {

            if (weight_flag) {
                sscanf(line, "%d %d %d", &head, &tail, &weight);
            } else {
                sscanf(line, "%d %d",  &head, &tail);
                printf("(%d, %d)\n", head, tail);
                weight = 0;
            }

            if (tail == head) {
                printf("reporting self loop\n");
                continue;
            };

            CooTuple temp;
            temp.row = head - 1;
            temp.col = tail - 1;
            temp.val = weight;
            tuple_array[cnt++] = temp;

            if (!directed) {
                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }

        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }
    row_array[row_cnt] = idx;

    fclose(fptr);
    free(tuple_array);

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    return csr;
}

csr_array *parseMetis_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char *line = (char *)malloc(8192);
    int num_edges = 0, num_nodes = 0;
    int *col_cnt = NULL;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);
    while (fgets(line, 8192, fptr)) {
        int head, tail, weight = 0;
        CooTuple temp;

        if (line[0] == '%') continue; // skip comment lines

        if (lineno == 0) { //the first line

            sscanf(line, "%d %d", p_num_nodes, p_num_edges);

            col_cnt = (int *)malloc(*p_num_nodes * sizeof(int));
            if (!col_cnt) {
                printf("memory allocation failed for col_cnt\n");
                exit(1);
            }
            memset(col_cnt, 0, *p_num_nodes * sizeof(int));

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }
            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
        } else if (lineno > 0) { //from the second line
            char *pch;
            pch = strtok(line , " ,.-");
            while (pch != NULL) {
                head = lineno;
                tail = atoi(pch);
                if (tail <= 0) {
                    break;
                }

                if (tail == head) printf("reporting self loop: %d, %d\n", lineno + 1, lineno);

                temp.row = tail - 1;
                temp.col = head - 1;
                temp.val = weight;

                col_cnt[head - 1]++;
                tuple_array[cnt++] = temp;

                pch = strtok(NULL, " ,.-");
            }
        }
#ifdef VERBOSE
        printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    printf("tohere\n"); fflush(stdout);
    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }
        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }
    row_array[row_cnt] = idx;

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;
    csr->col_cnt = col_cnt;
    fclose(fptr);

    return csr;
}

csr_array *parseCOO_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed)
{
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
    int num_nodes = 0, num_edges = 0;

    FILE *fptr;
    CooTuple *tuple_array = NULL;

    fptr = fopen(tmpchar, "r");
    if (!fptr) {
        fprintf(stderr, "Error when opening file: %s\n", tmpchar);
        exit(1);
    }

    printf("Opening file: %s\n", tmpchar);

    while (fgets(line, 100, fptr)) {
        int head, tail, weight;
        switch (line[0]) {
        case 'c':
            break;
        case 'p':
            fflush(stdout);

            sscanf(line, "%c %s %d %d", &p, sp, p_num_nodes, p_num_edges);

            if (!directed) {
                *p_num_edges = *p_num_edges * 2;
                printf("This is an undirected graph\n");
            } else {
                printf("This is a directed graph\n");
            }

            num_nodes = *p_num_nodes;
            num_edges = *p_num_edges;

            printf("Read from file: num_nodes = %d, num_edges = %d\n", num_nodes, num_edges);
            tuple_array = (CooTuple *)malloc(sizeof(CooTuple) * num_edges);
            break;

        case 'a':
            sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
            if (tail == head) printf("reporting self loop\n");
            CooTuple temp;
            temp.val = weight;
            temp.row = tail - 1;
            temp.col = head - 1;
            tuple_array[cnt++] = temp;
            if (!directed) {
                temp.val = weight;
                temp.row = tail - 1;
                temp.col = head - 1;
                tuple_array[cnt++] = temp;
            }

#ifdef VERBOSE
            printf("Adding edge: %d ==> %d ( %d )\n", head, tail, weight);
#endif
            break;
        default:
            fprintf(stderr, "exiting loop\n");
            break;
        }
        lineno++;
    }

    std::stable_sort(tuple_array, tuple_array + num_edges, doCompare);

#ifdef VERBOSE
    for (int i = 0 ; i < num_edges; i++) {
        printf("%d: %d, %d, %d\n", i, tuple_array[i].row, tuple_array[i].col, tuple_array[i].val);
    }
#endif

    int *row_array = (int *)malloc((num_nodes + 1) * sizeof(int));
    int *col_array = (int *)malloc(num_edges * sizeof(int));
    int *data_array = (int *)malloc(num_edges * sizeof(int));

    int row_cnt = 0;
    int prev = -1;
    int idx;
    for (idx = 0; idx < num_edges; idx++) {
        int curr = tuple_array[idx].row;
        if (curr != prev) {
            row_array[row_cnt++] = idx;
            prev = curr;
        }
        col_array[idx] = tuple_array[idx].col;
        data_array[idx] = tuple_array[idx].val;
    }
    while (row_cnt <= num_nodes) {
        row_array[row_cnt++] = idx;
    }

    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
//    memset(csr, 0, sizeof(csr_array));
    csr->row_array = row_array;
    csr->col_array = col_array;
    csr->data_array = data_array;

    fclose(fptr);
    free(tuple_array);

    return csr;
}


