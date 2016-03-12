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
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
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
#include <algorithm>
#include "BC.h"
#include "../../graph_parser/util.h"

int initialize(int use_gpu);
int shutdown();

void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

// local variables
static cl_context	    context;
static cl_command_queue   cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

int main(int argc, char **argv){
	
    char *tmpchar;	
    char *filechar;

    int num_nodes; 
    int num_edges;
    int use_gpu = 1;
    int file_format = 1;
    bool directed = 1;

    cl_int err = 0;

    if(argc == 4){
       tmpchar     = argv[1];       //graph inputfile
       filechar    = argv[2];	    //kernel file
       file_format = atoi(argv[3]); //file format
    }
    else{
       fprintf(stderr, "You did something wrong!\n");
       exit(1);
    }

	//allocate the csr structure
    csr_array *csr  = (csr_array *)malloc(sizeof(csr_array));
	if(!csr) fprintf(stderr, "malloc failed csr\n");
	
	//parse graph and store it in a CSR format
    csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);

	//allocate the bc host array
    float *bc_h = (float *)malloc(num_nodes * sizeof(float));
    if(!bc_h) fprintf(stderr, "malloc failed bc_h\n");

    //load kernel file
    int sourcesize = 1024*1024;
    char * source = (char *)calloc(sourcesize, sizeof(char)); 
    if(!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

    FILE * fp = fopen(filechar, "rb"); 
    if(!fp) { fprintf(stderr, "ERROR: unable to open '%s'\n", filechar); return -1; }
    fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    // OpenCL initialization
    if(initialize(use_gpu)) return -1;

    // Create OpenCL program
    const char * slist[2] = { source, 0 };
    cl_program prog = clCreateProgramWithSource(context, 
                                                1, 
                                                slist, 
                                                NULL, 
                                                &err);
												
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    { // show warnings/errors
        static char log[65536]; 
		memset(log, 0, sizeof(log));
        cl_device_id device_id = 0;
		
        err = clGetContextInfo( context, 
                                CL_CONTEXT_DEVICES,
                                sizeof(device_id), 
                                &device_id, 
                                NULL);
							   
        clGetProgramBuildInfo( prog, 
                               device_id, 
                               CL_PROGRAM_BUILD_LOG, 
                               sizeof(log)-1, 
                               log, 
                               NULL);
							   
        if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    }

    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clBuildProgram() => %d\n", err); return -1; }
    	
    cl_kernel kernel1, kernel2, kernel3, kernel4, kernel5;

    char * kernelbc1  = "bfs_kernel";
    char * kernelbc2  = "backtrack_kernel";
    char * kernelbc3  = "clean_1d_array"; 
    char * kernelbc4  = "clean_2d_array"; 
    char * kernelbc5  = "clean_bc";

	// Create OpenCL kernels
    kernel1 = clCreateKernel(prog, kernelbc1, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 1 => %d\n", err); return -1; }
    kernel2 = clCreateKernel(prog, kernelbc2, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 2 => %d\n", err); return -1; }
    kernel3 = clCreateKernel(prog, kernelbc3, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 3 => %d\n", err); return -1; }
    kernel4 = clCreateKernel(prog, kernelbc4, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 4 => %d\n", err); return -1; }
    kernel5 = clCreateKernel(prog, kernelbc5, &err);  
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 5 => %d\n", err); return -1; }

    clReleaseProgram(prog);

    //create device-side buffers
    cl_mem bc_d, dist_d, sigma_d, rho_d, p_d, stop_d;
    cl_mem row_d, col_d, row_trans_d, col_trans_d;

	//create betweenness centrality buffers
    bc_d = clCreateBuffer( context, CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err ); 
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer bc_d (size:%d) => %d\n",  num_nodes , err); return -1;}	
    dist_d = clCreateBuffer( context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer dist_d (size:%d) => %d\n",  num_nodes , err); return -1;}
    sigma_d = clCreateBuffer( context,CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer sigma_d (size:%d) => %d\n", num_nodes , err); return -1;}	
    rho_d = clCreateBuffer( context, CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer rho_d (size:%d) => %d\n",  num_nodes , err); return -1;}
    p_d = clCreateBuffer( context,CL_MEM_READ_WRITE, num_nodes * num_nodes * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer p_d (size:%d) => %d\n", num_nodes * num_nodes , err); return -1;}	

	//create termination variable buffer
    stop_d = clCreateBuffer(context,CL_MEM_READ_WRITE, sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer stop_d (size:%d) => %d\n", 1 , err); return -1;}	

	//create graph buffers
    row_d = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes + 1) * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer row_d (size:%d) => %d\n",  num_nodes , err); return -1;}	
    col_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_edges * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer col_d (size:%d) => %d\n",  num_edges , err); return -1;}
    row_trans_d = clCreateBuffer(context, CL_MEM_READ_WRITE, (num_nodes + 1) * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer row_d_t (size:%d) => %d\n",  num_nodes , err); return -1;}	
    col_trans_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_edges * sizeof(int), NULL, &err );
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer col_d_t (size:%d) => %d\n",  num_edges , err); return -1;}    

    double timer1, timer2;
    double timer3, timer4;

    timer1=gettime();

    //copy data to device-side buffers
    err = clEnqueueWriteBuffer(cmd_queue, 
                               row_d, 
                               1, 
                               0, 
                               (num_nodes + 1) * sizeof(int), 
                               csr->row_array, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer row_d (size:%d) => %d\n", num_nodes, err); return -1; }

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
    
    //copy data to device-side buffers
    err = clEnqueueWriteBuffer(cmd_queue, 
                               row_trans_d, 
                               1, 
                               0, 
                               (num_nodes + 1) * sizeof(int), 
                               csr->row_array_t, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer row_trans_d (size:%d) => %d\n", num_nodes, err); return -1; }
	
    err = clEnqueueWriteBuffer(cmd_queue, 
                               col_trans_d, 
                               1, 
                               0, 
                               num_edges * sizeof(int), 
                               csr->col_array_t, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer col_trans_d (size:%d) => %d\n", num_nodes, err); return -1; }

	//set up kernel arguments
    //kernel1
    clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &row_d);
    clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &col_d);
    clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &dist_d);	
    clSetKernelArg(kernel1, 3, sizeof(void *), (void*) &rho_d);
    clSetKernelArg(kernel1, 4, sizeof(void *), (void*) &p_d);
    clSetKernelArg(kernel1, 5, sizeof(void *), (void*) &stop_d);
    clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &num_nodes);	
    clSetKernelArg(kernel1, 7, sizeof(cl_int), (void*) &num_edges);

    //kernel2
    clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &row_trans_d);
    clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &col_trans_d);
    clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &dist_d);
    clSetKernelArg(kernel2, 3, sizeof(void *), (void*) &rho_d);
    clSetKernelArg(kernel2, 4, sizeof(void *), (void*) &sigma_d);
    clSetKernelArg(kernel2, 5, sizeof(void *), (void*) &p_d);
    clSetKernelArg(kernel2, 6, sizeof(cl_int), (void*) &num_nodes);	
    clSetKernelArg(kernel2, 7, sizeof(cl_int), (void*) &num_edges);
    clSetKernelArg(kernel2, 10, sizeof(void *), (void*) &bc_d);

    //kernel3
    clSetKernelArg(kernel3, 1, sizeof(void *), (void*) &dist_d);
    clSetKernelArg(kernel3, 2, sizeof(void *), (void*) &sigma_d);
    clSetKernelArg(kernel3, 3, sizeof(void *), (void*) &rho_d);
    clSetKernelArg(kernel3, 4, sizeof(cl_int), (void*) &num_nodes);	

    //kernel4
    clSetKernelArg(kernel4, 0, sizeof(void *), (void*) &p_d);
    clSetKernelArg(kernel4, 1, sizeof(cl_int), (void*) &num_nodes);	

    //kernel5
    clSetKernelArg(kernel5, 0, sizeof(void *), (void*) &bc_d);
    clSetKernelArg(kernel5, 1, sizeof(cl_int), (void*) &num_nodes);

    timer3=gettime();

    //set up kernel dimensions
    int local_worksize = 128;
    size_t local_work[3] = { local_worksize,  1, 1};
    size_t global_work[3] = { (num_nodes%local_worksize == 0)? num_nodes: (num_nodes/local_worksize + 1) * local_worksize, 1,  1 }; 

    //initialization
    err = clEnqueueNDRangeKernel(cmd_queue, 
                                 kernel5, 
                                 1, 
                                 NULL, 
                                 global_work, 
                                 local_work, 
                                 0, 
                                 0, 
                                 0);
								 
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
	
    // main computation loop
    for(int i = 0; i < num_nodes; i++){

        size_t local_work[3] =  { local_worksize,  1, 1};
        size_t global_work[3] = { (num_nodes%local_worksize == 0)? num_nodes: (num_nodes/local_worksize + 1) * local_worksize, 1,  1 }; 

        clSetKernelArg(kernel3, 0, sizeof(cl_int), (void*) &i);
		
		//clean up the sigma, distance, and rho arrays
        err = clEnqueueNDRangeKernel(cmd_queue, 
                                     kernel3, 
                                     1, 
                                     NULL, 
                                     global_work, 
                                     local_work, 
                                     0, 
                                     0, 
                                     0);
									 
        if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

        global_work[0] = ((num_nodes * num_nodes) %local_worksize == 0)? num_nodes * num_nodes: ((num_nodes * num_nodes)/local_worksize + 1) * local_worksize;
		
        //clean up the p array for tracking dependencies
        err = clEnqueueNDRangeKernel(cmd_queue, 
                                     kernel4, 
                                     1, 
                                     NULL, 
                                     global_work, 
                                     local_work, 
                                     0, 
                                     0, 
                                     0);
									 
        if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

        // depth of the traversal
        int dist = 0;
		// termination variable
        int stop = 1; 

        //traverse the graph from the source node i
        do {
            stop = 0;
			
			//copy the termination variable to the device
            err = clEnqueueWriteBuffer(cmd_queue, stop_d, 1, 0, sizeof(int), &stop, 0, 0, 0);
			
            global_work[0] = (num_nodes%local_worksize == 0)? num_nodes: (num_nodes/local_worksize + 1) * local_worksize;  
            clSetKernelArg(kernel1, 8, sizeof(cl_int), (void*) &dist);
            
            //launch the breadth first traversal kernel
			err = clEnqueueNDRangeKernel(cmd_queue, 
                                         kernel1, 
                                         1, 
                                         NULL, 
                                         global_work, 
                                         local_work, 
                                         0, 
                                         0, 
                                         0);
										 
            if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel1 (%d)\n", err); return -1; }
			
			//copy back the termination variable from the device
            err = clEnqueueReadBuffer(cmd_queue, stop_d, 1, 0, sizeof(int), &stop, 0, 0, 0);
			
			//another level
            dist++;

        } while(stop) ;

        clFinish(cmd_queue);

        //traverse back from the deepest part of the tree
        while(dist){

	        global_work[0] = (num_nodes%local_worksize == 0)? num_nodes: (num_nodes/local_worksize + 1) * local_worksize;

            clSetKernelArg(kernel2, 8, sizeof(cl_int), (void*) &dist);	
            clSetKernelArg(kernel2, 9, sizeof(cl_int), (void*) &i);	
			
            //launch the kernel to back traverse to the source node i
            err = clEnqueueNDRangeKernel(cmd_queue, 
                                         kernel2, 
                                         1, 
                                         NULL, 
                                         global_work, 
                                         local_work, 
                                         0, 
                                         0, 
                                         0);
            if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel2 (%d)\n", err); return -1; }
	   
            //back one level
            dist--;  
        }
        clFinish(cmd_queue);

    }
    clFinish(cmd_queue);
    timer4=gettime();

	//copy back the results for the bc array
    err = clEnqueueReadBuffer(cmd_queue, 
                              bc_d, 
                              1, 
                              0, 
                              num_nodes * sizeof(float), 
                              bc_h, 
                              0, 
                              0, 
                              0);		
							  
    if(err != CL_SUCCESS) { printf("ERROR: read buffer bc_d (%d)\n", err); return -1; }
	
    timer2=gettime();
	
    printf("kernel + memcopy time = %lf ms\n",(timer4-timer3)*1000);
    printf("kernel execution time = %lf ms\n",(timer2-timer1)*1000);

#if 0 
    //dump the results to the file
    print_vectorf(bc_h, num_nodes);
#endif

    //clean up the host-side buffers
    free(bc_h);
    free(csr->row_array);
    free(csr->col_array);
    free(csr->data_array);
    free(csr->row_array_t);
    free(csr->col_array_t);
    free(csr->data_array_t);
    free(csr);

	//clean up the device-side buffers
    clReleaseMemObject(bc_d);
    clReleaseMemObject(dist_d); 
    clReleaseMemObject(sigma_d); 
    clReleaseMemObject(rho_d); 
    clReleaseMemObject(p_d);
    clReleaseMemObject(stop_d);
    clReleaseMemObject(row_d); 
    clReleaseMemObject(col_d);
    clReleaseMemObject(row_trans_d);
    clReleaseMemObject(col_trans_d);

    //clean up the OpenCL variables
    shutdown();

    return 0;

}

void print_vector(int *vector, int num){
    for(int i = 0; i < num; i++)
       printf("%d: %d \n",i+1, vector[i]);
    printf("\n");
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
									   
    if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

    // get the list of GPUs
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
    num_devices = (int) (size / sizeof(cl_device_id));
    printf("num_devices = %d\n", num_devices);

    if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
    device_list = new cl_device_id[num_devices];
    if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
    if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

    // create command queue for the first device
    cmd_queue = clCreateCommandQueue( context, 
                                      device_list[0], 
                                      0, 
                                      NULL );
									  
    if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
    return 0;
}

int shutdown()
{
    // release resources
    if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
    if( context )   clReleaseContext( context );
    if( device_list ) delete[] device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;

}
