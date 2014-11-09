/************************************************************************************\ 
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *   
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      * 
 *                                                                                  *  
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  * 
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *  
 * technologies for which you must obtain licenses from parties other than AMD.     *  
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  * 
 * underlying intellectual property rights related to the third party technologies. *  
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *        
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR"ù) (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    * 
 * E:2 any restricted technology, software, or source code you receive hereunder,   * 
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    * 
 * national security controls as identified on the Commerce Control List (currently * 
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   * 
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <CL/cl.h>
#include <omp.h>
#include "../../graph_parser/util.h"

#define BIGNUM 999999
#define TRUE 1
#define FALSE 0

int initialize(int use_gpu);
int shutdown();

void set_value(int* array, int dim, int i, int j, int value);
void dump2file(int *adjmatrix, int num_nodes);
void print_vector(int *vector, int num);
int* parse_graph_file(int *num_nodes, int *num_edges, char* tmpchar);
void adjmatrix2CSR(int *adjmatrix, int *row_array, int *col_array, int *data_array, int num_nodes, int num_edges);
bool test_value(int* array, int dim, int i, int j);

// OpenCL  variables
static cl_context	context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

int main(int argc, char **argv){
	
    char *tmpchar;	
    char *filechar;

    int num_nodes; 
    int num_edges;
    int use_gpu = 1;

    cl_int err = 0;

	//get program input
    if(argc == 3){
       tmpchar = argv[1];  //graph input file
       filechar = argv[2]; //kernel file
    }
    else{
       fprintf(stderr, "You did something wrong!\n");
       exit(1);
    }	

	//parse the adjacency matrix
    int *adjmatrix = parse_graph_file(&num_nodes, &num_edges, tmpchar);	
    int dim = num_nodes;

	//initialize the distance matrix
    int *distmatrix = (int *)malloc(dim * dim * sizeof(int));
    if(!distmatrix) fprintf(stderr, "malloc failed - distmatrix\n");
	
	//initialize the result matrix 
    int *result     = (int *)malloc(dim * dim * sizeof(int));
    if(!result) fprintf(stderr, "malloc failed - result\n");
	
	//initialize the result matrix on the CPU 
    int *result_cpu = (int *)malloc(dim * dim * sizeof(int));
    if(!result_cpu) fprintf(stderr, "malloc failed - result_cpu\n");

    //TODO: now only supports integer weights
	//setup the input matrix
    for (int i = 0 ; i < dim; i++){
        for (int j = 0 ; j < dim; j++){
             //diagonal 
             if (i == j)
                distmatrix[i * dim + j] = 0;			
             //without edge
             else if (adjmatrix[i * dim + j] == -1 )	
                distmatrix[i * dim + j] = BIGNUM;
             //with edge
             else	
                distmatrix[i * dim + j] = adjmatrix[i * dim + j];
        }
    }

	//load the OpenCL kernel
    int sourcesize = 1024*1024;
    char * source = (char *)calloc(sourcesize, sizeof(char)); 
    if(!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

    FILE * fp = fopen(filechar, "rb"); 
    if(!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", filechar); return -1; }
    fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    // OpenCL initialization
    if(initialize(use_gpu)) return -1;

	//create the OpenCL program
    const char * slist[2] = { source, 0 };
    cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    { // show warnings/errors
	  static char log[65536]; 
      memset(log, 0, sizeof(log));
 	  cl_device_id device_id = 0;
	  //get context info
      err = clGetContextInfo(context, 
                             CL_CONTEXT_DEVICES, 
                             sizeof(device_id), 
                             &device_id, 
                             NULL);
      //get build info					 
      clGetProgramBuildInfo(prog, 
                            device_id, 
                            CL_PROGRAM_BUILD_LOG, 
                            sizeof(log)-1, 
                            log, 
                            NULL);
							
	  if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    }
	
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clBuildProgram() => %d\n", err); return -1; }

    //create the OpenCL kernel 
    cl_kernel kernel1;
    char * kernelfw  = "floydwarshall";
    kernel1 = clCreateKernel(prog, kernelfw, &err);  

    if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
    clReleaseProgram(prog);

    cl_mem dist_d;
    cl_mem next_d;

    //create device-side FW buffers
    dist_d = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer dist_d (size:%d) => %d\n",  dim * dim , err); return -1;}	
    next_d = clCreateBuffer(context, CL_MEM_READ_WRITE, dim * dim * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer next_d (size:%d) => %d\n",  dim * dim , err); return -1;}

    double timer1=gettime();

	//copy the dist matrix to the device
    err = clEnqueueWriteBuffer(cmd_queue, 
                               dist_d, 
                               1, 
                               0, 
                               dim * dim * sizeof(int), 
                               distmatrix, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer feature_d (size:%d) => %d\n", dim * dim, err); return -1; }

    //OpenCL work dimension
    size_t local_work[3] =  { 16,  16, 1};
    size_t global_work[3] = { num_nodes, num_nodes, 1 }; 

	//set up kernel arguments
    clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &dist_d);
    clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &next_d);
    clSetKernelArg(kernel1, 2, sizeof(cl_int), (void*) &dim);

    double timer3 = gettime();
    //main computation loop
    for(int k = 1; k < dim; k++){
	
        clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &k);
		
		//launch the naive floydwarhall kernel
        err = clEnqueueNDRangeKernel(cmd_queue, 
                                     kernel1, 
                                     2, 
                                     NULL, 
                                     global_work, 
                                     local_work, 
                                     0, 
                                     0, 
                                     0);
									 
        if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
    }
    clFinish(cmd_queue);

    double timer4 = gettime();
    err = clEnqueueReadBuffer(cmd_queue, 
                              dist_d, 
                              1, 
                              0, 
                              dim * dim * sizeof(int), 
                              result, 
                              0, 
                              0, 
                              0);
	
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR:  read back dist_d %d failed\n", err); return -1; }
    double timer2=gettime();

    printf("kernel time = %lf ms\n",(timer4-timer3)*1000);
    printf("kernel + memcpy time = %lf ms\n",(timer2-timer1)*1000);

    //below is the verification part
    //caluate on the CPU
    int *dist = distmatrix;
    for(int k = 0; k < dim; k++){
        for(int i = 0; i < dim; i++){
            for(int j = 0; j < dim; j++){
               if (dist[i * dim + k] + dist[k * dim + j] < dist[i * dim + j]){
                   dist[i * dim + j] = dist[i * dim + k] + dist[k * dim + j];
	       }
	    }
	}
    }

	//compare results
    bool check_flag = 0;
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < dim; j++){
           if (dist[i * dim + j] !=  result[i * dim + j]){
               printf("mismatch at (%d, %d)", i, j);
               check_flag = 1;
	       }
	    }
    }
	//if there is mismatch, report
    if(check_flag) fprintf(stderr, "produce wrong results\n");

    printf("Finishing Floyd-Warshall\n");

    //free host-side buffers
    free(adjmatrix);
    free(result);
    free(distmatrix);

    //free OpenCL buffers
    clReleaseMemObject(dist_d);
    clReleaseMemObject(next_d);	

    //clean up OpenCL buffers
    shutdown();

    return 0;

}

//2D adjacency matrix -> CSR array
void adjmatrix2CSR(int* adjmatrix, int *row_array, int *col_array, int *data_array, int num_nodes, int num_edges){

    int col_cnt = 0;
    int row_cnt = 0;

    bool first;
    for(int i = 0; i < num_nodes; i++){
        first = FALSE;
        for(int j = 0; j < num_nodes; j++){
            if (adjmatrix[i * num_nodes+j]!= -1){
                col_array[col_cnt] = j;
                data_array[col_cnt] = adjmatrix[i * num_nodes +j];
                if (first == FALSE){
                   row_array[row_cnt++] = col_cnt + 1;
	               first = TRUE;
                } 
            col_cnt++;
            }
         }
    }
    row_array[row_cnt] = num_edges;	
 	
}

int* parse_graph_file(int *num_nodes, int *num_edges, char* tmpchar){

    int *adjmatrix;
    int cnt = 0;
    unsigned int lineno = 0;
    char line[128], sp[2], a, p;
	
    FILE *fptr;

    fptr = fopen(tmpchar, "r");
    
    if(!fptr) {
       fprintf(stderr, "Error when opennning file: %s\n", tmpchar);
       exit(1);
    }
 
    printf("Opening file: %s\n", tmpchar);

    while(fgets(line, 100, fptr))
    {
         int head, tail, weight, size;
         switch(line[0])
         {
             case 'c':
                      break;
             case 'p':
                      sscanf(line, "%c %s %d %d", &p, sp, num_nodes, num_edges);
                      printf("Read from file: num_nodes = %d, num_edges = %d\n", *num_nodes, *num_edges);
                      size = (*num_nodes + 1) * (*num_nodes + 1); 
                      adjmatrix = (int *)malloc(size * sizeof(int));
                      memset(adjmatrix, -1 , size * sizeof(int));	
                      break;
             case 'a':
                      sscanf(line, "%c %d %d %d", &a, &head, &tail, &weight);
                      if (tail == head) printf("reporting self loop\n");
                      if (test_value(adjmatrix, *num_nodes + 1, head, tail)){
                         set_value(adjmatrix, *num_nodes + 1, head, tail, weight);
                         cnt++;
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

    *num_edges = cnt;	
    printf("Actual added edges: %d\n", cnt);	

    fclose(fptr);

    return adjmatrix;
	
}

void dump2file(int *adjmatrix, int num_nodes){

    FILE *dptr; 	

    printf("Dumping the adjacency matrix to adjmatrix_dump.txt\n");
    dptr = fopen("adjmatrix_dump.txt", "w");	
    for(int i=0 ; i <num_nodes; i++){
        for(int j =0; j < num_nodes; j++){
            fprintf(dptr, "%d ", adjmatrix[i*num_nodes+j]);
        }
        fprintf(dptr, "\n");
    }
    fclose(dptr);
}

//print the values
void print_vector(int *vector, int num){
    for(int i = 0; i < num; i++){
        printf("%d ", vector[i]);
    }
    printf("\n");
}

//test value
bool test_value(int* array, int dim, int i, int j){

    //TODO: current does not support multiple edges between two vertices
    if (array[i * dim + j] != -1) {
        //fprintf(stderr, "Possibly duplicate records at (%d, %d)\n", i, j);
        return 0;
    }
    else 
        return 1;
}

//set value (i, j) = value
void set_value(int* array, int dim, int i, int j, int value){
    array[i * dim + j] = value;
}

int initialize(int use_gpu)
{
    cl_int result;
    size_t size;

    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
    cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
	
    context = clCreateContextFromType( ctxprop, 
                                       device_type, 
                                       NULL, 
                                       NULL, 
                                       NULL );
									   
    if( !context ) { fprintf(stderr, "ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

    // get the list of GPUs
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
    num_devices = (int) (size / sizeof(cl_device_id));
    printf("num_devices = %d\n", num_devices);
    if( result != CL_SUCCESS || num_devices < 1 ) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
	
    device_list = new cl_device_id[num_devices];
    if( !device_list ) { fprintf(stderr, "ERROR: new cl_device_id[] failed\n"); return -1; }
	
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
    if( result != CL_SUCCESS ) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }

    // create command queue for the first device
    cmd_queue = clCreateCommandQueue( context, 
                                      device_list[0], 
                                      0, 
                                      NULL );
									  
    if( !cmd_queue ) { fprintf(stderr, "ERROR: clCreateCommandQueue() failed\n"); return -1; }
	
    return 0;
}

int shutdown()
{
    // release resources
    if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
    if( context ) clReleaseContext( context );
    if( device_list ) delete device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;

}
