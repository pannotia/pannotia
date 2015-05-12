#include <stdlib.h>

typedef struct csr_arrays_t {
    int *row_array;
    int *col_array;
    int *data_array;
    int *col_cnt;

    void freeArrays() {
        if (row_array) {
            free(row_array);
            row_array = NULL;
        }
        if (col_array) {
            free(col_array);
            col_array = NULL;
        }
        if (data_array) {
            free(data_array);
            data_array = NULL;
        }
        if (col_cnt) {
            free(col_cnt);
            col_cnt = NULL;
        }
    }
} csr_array;

typedef struct ell_arrays_t {
    int max_height;
    int num_nodes;
    int *col_array;
    int *data_array;
//  int *col_cnt;
} ell_array;

typedef struct double_edges_t {
    int *edge_array1;
    int *edge_array2;
} double_edges;

typedef struct cooedgetuple {
    int row;
    int col;
    int val;
} CooTuple;

csr_array *parseCOO(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
csr_array *parseMetis(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
csr_array *parseMM(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed, bool weight_flag);
ell_array *csr2ell(csr_array *csr, int num_nodes, int num_edges, int fill);

double_edges *parseCOO_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
double_edges *parseMetis_doubleEdge(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);

csr_array *parseCOO_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
csr_array *parseMetis_transpose(char* tmpchar, int *p_num_nodes, int *p_num_edges, bool directed);
