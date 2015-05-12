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
#include "../../graph_parser/parse.h"
#include "../../graph_parser/util.h"

//iteration count
#define ITER 20

int initialize(int use_gpu);
int shutdown();
void print_vectorf(float *vector, int num);

//OpenCL variables 
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

int main(int argc, char **argv){
	
    char *tmpchar;
    char *filechar;

    int num_nodes; 
    int num_edges;
    int use_gpu = 0;
    int file_format = 1;
    bool directed = 0;

    cl_int err = 0;

    if (argc == 4){
        tmpchar = argv[1];            //graph inputfile
        filechar = argv[2];	          //kernel inputfile
        file_format = atoi(argv[3]);  //file format
    }
    else{
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    //allocate the csr structure
    csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
	if(!csr){ fprintf(stderr, "ERROR: malloc csr failed\n"); return -1; }

    //parse graph files into csr structure 
    if (file_format == 1) //metis
        csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
    else if (file_format == 0) //dimacs9		
        csr = parseCOO(tmpchar, &num_nodes, &num_edges, 1);
    else if (file_format == 2) //matrix market
        csr = parseMM(tmpchar, &num_nodes, &num_edges, directed, 0);
    else{
        printf("reserve for future");
        exit(1);
    }
	
	//allocate rank_array
    float *rank_array = (float *)malloc(num_nodes * sizeof(float));
    if(!rank_array) {fprintf(stderr, "rank array not allocated successfully\n"); return -1;} 

    //load the OpenCL kernel files
    int sourcesize = 1024*1024;
    char * source = (char *)calloc(sourcesize, sizeof(char)); 
    if(!source) {fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

    FILE * fp = fopen(filechar, "rb"); 
    if(!fp) {fprintf(stderr, "ERROR: unable to open '%s'\n", filechar); return -1; }
    fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    //initialize OpenCL variables
    if(initialize(use_gpu)) return -1;

	//create the OpenCL program
    const char * slist[2] = { source, 0 };
    cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    { 
        static char log[65536]; 
        memset(log, 0, sizeof(log));
        cl_device_id device_id = 0;
		
        err = clGetContextInfo(context, 
                               CL_CONTEXT_DEVICES, 
                               sizeof(device_id), 
                               &device_id, 
                               NULL);
							   
        clGetProgramBuildInfo(prog, 
                              device_id, 
                              CL_PROGRAM_BUILD_LOG, 
                              sizeof(log)-1, 
                              log, 
                              NULL);
							  
        if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    }
    
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clBuildProgram() => %d\n", err); return -1; }

    //create OpenCL kernels 
    cl_kernel kernel1, kernel2, kernel3;
    char * kernelpr1  = "pagerank1";
    char * kernelpr2  = "inibuffer";
    char * kernelpr3  = "pagerank2";

    kernel1 = clCreateKernel(prog, kernelpr1, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 1 => %d\n", err); return -1; }
    kernel2 = clCreateKernel(prog, kernelpr2, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 2 => %d\n", err); return -1; }
    kernel3 = clCreateKernel(prog, kernelpr3, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 3 => %d\n", err); return -1; }

    clReleaseProgram(prog);

    cl_mem row_d;
    cl_mem col_d;
    cl_mem data_d;

    cl_mem pagerank1_d;
    cl_mem pagerank2_d;

    //create OpenCL device-side buffers for the graph
    row_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer row_d (size:%d) => %d\n",  num_nodes , err); return -1;}	
    col_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_edges * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer col_d (size:%d) => %d\n",  num_edges , err); return -1;}
    data_d = clCreateBuffer(context,CL_MEM_READ_WRITE, num_edges * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer data_d (size:%d) => %d\n", num_edges , err); return -1;}	

    //create OpenCL buffers for pagerank
    pagerank1_d = clCreateBuffer(context,CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer mask_d (size:%d) => %d\n", num_nodes , err); return -1;}	
    pagerank2_d = clCreateBuffer(context,CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer C_d (size:%d) => %d\n", num_nodes , err); return -1;}	

    double timer1=gettime();

	//copy the data to the device-side buffers
    err = clEnqueueWriteBuffer(cmd_queue, 
                               row_d, 
                               1, 
                               0, 
                               num_nodes * sizeof(int), 
                               csr->row_array, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR:#endif clEnqueueWriteBuffer row_d (size:%d) => %d\n", num_nodes, err); return -1; }

    err = clEnqueueWriteBuffer(cmd_queue, 
                               col_d, 
                               1, 
                               0, 
                               num_edges * sizeof(int), 
                               csr->col_array, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer col_d (size:%d) => %d\n", num_nodes, err); return -1; }	

	//set up OpenCL work dimensions
    int block_size  = 256;
    int global_size = (num_nodes%block_size == 0)? num_nodes: (num_nodes/block_size + 1) * block_size;

    size_t local_work[3] =  { block_size,  1, 1};
    size_t global_work[3] = { global_size, 1, 1}; 

	//set up kernel arguments
    clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &row_d);
    clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &pagerank1_d);
    clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &pagerank2_d);
    clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &num_nodes);
    clSetKernelArg(kernel2, 4, sizeof(cl_int), (void*) &num_edges);

    double timer3=gettime();
    //launch the initialization kernel
    err = clEnqueueNDRangeKernel(cmd_queue, 
                                 kernel2, 
                                 1, 
                                 NULL, 
                                 global_work, 
                                 local_work, 
                                 0, 
                                 0, 
                                 0);
								 
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

    //set up kernel arguments
	//kernel 1
    clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &row_d);
    clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &col_d);
    clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &data_d);
    clSetKernelArg(kernel1, 3, sizeof(void *), (void*) &pagerank1_d);
    clSetKernelArg(kernel1, 4, sizeof(void *), (void*) &pagerank2_d);
    clSetKernelArg(kernel1, 5, sizeof(cl_int), (void*) &num_nodes);
    clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &num_edges);

    //kernel 3
    clSetKernelArg(kernel3, 0, sizeof(void *), (void*) &row_d);
    clSetKernelArg(kernel3, 1, sizeof(void *), (void*) &col_d);
    clSetKernelArg(kernel3, 2, sizeof(void *), (void*) &data_d);
    clSetKernelArg(kernel3, 3, sizeof(void *), (void*) &pagerank1_d);
    clSetKernelArg(kernel3, 4, sizeof(void *), (void*) &pagerank2_d);
    clSetKernelArg(kernel3, 5, sizeof(cl_int), (void*) &num_nodes);
    clSetKernelArg(kernel3, 6, sizeof(cl_int), (void*) &num_edges);

    //run PageRank for some iter. TO: convergence determination
    for(int i = 0; i < ITER ; i++){
	    //launch the pagerank kernel 1
        err = clEnqueueNDRangeKernel(cmd_queue, 
                                     kernel1, 
                                     1, 
                                     NULL, 
                                     global_work, 
                                     local_work, 
                                     0, 
                                     0, 
                                     0);
									 
        if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
        //launch the pagerank kernel 2	
        err = clEnqueueNDRangeKernel(cmd_queue, 
                                     kernel3, 
                                     1, 
                                     NULL, 
                                     global_work, 
                                     local_work, 
                                     0, 
                                     0, 
                                     0);
									 
        if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
    }
    clFinish(cmd_queue);

    double timer4=gettime();

    //copy the rank buffere back
    err = clEnqueueReadBuffer(cmd_queue, 
                              pagerank1_d, 
                              1, 
                              0, 
                              num_nodes * sizeof(float), 
                              rank_array, 
                              0, 
                              0, 
                              0);
							  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueReadBuffer()=>%d failed\n", err); return -1; }

    double timer2=gettime();

	//report timing characteristics 
    printf("kernel time          = %lf ms\n",(timer4-timer3)*1000);
    printf("kernel + memcpy time = %lf ms\n",(timer2-timer1)*1000);

#if 1
    //print out some results
    //double sum = 0;
    //for(int i = 0; i < num_nodes; i++){
    //    sum += rank_array[i];
    //}

    //print rank array
    print_vectorf(rank_array, num_nodes);
    //printf("sum = %f\n", sum);

#endif

    //free the host-side arrays
    free(rank_array);
    csr->freeArrays();
    free(csr);

    //free the OpenCL buffers 
    clReleaseMemObject(row_d);
    clReleaseMemObject(col_d);
    clReleaseMemObject(data_d);
	
    clReleaseMemObject(pagerank1_d);
    clReleaseMemObject(pagerank2_d);

    //clean up OpenCL variables 
    shutdown();

    return 0;

}

void print_vectorf(float *vector, int num){

    FILE * fp = fopen("result.out", "w"); 
    if(!fp) { printf("ERROR: unable to open result.txt\n");}

    for(int i = 0; i < num; i++){
        fprintf(fp, "%f\n", vector[i]);
    }

    fclose(fp);

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

