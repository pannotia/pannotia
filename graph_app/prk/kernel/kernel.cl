/************************************************************************************\
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
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



/**
 * @brief   atomic add float
 * @param   address     address
 * @param   value       value to be added
 */

inline float add_float_atomic(__global float* const address, 
                                       const  float value)
{
    uint oldval, newval, readback;
  
    *(float*)&oldval = *address;
    *(float*)&newval = (*(float*)&oldval + value);

    while ((readback = atomic_cmpxchg((__global uint*)address, oldval, newval)) != oldval) {
      oldval = readback;
      *(float*)&newval = (*(float*)&oldval + value);
    }

    return *(float*)&oldval;
}

/**
 * @brief   pagerank 1
 * @param   row         csr pointer array 
 * @param   col         csr column array 
 * @param   data        weight array 
 * @param   page_rank1  pagerank array 1
 * @param   page_rank2  pagerank array 2
 * @param   num_nodes   number of vertices
 * @param   num_edges   number of edges
 */
__kernel  void pagerank1(__global int *row, 
                         __global int *col, 
                         __global int *data,
                         __global float *page_rank1,
                         __global float *page_rank2,
                           const  int num_nodes, 
                           const  int num_edges )
{

    //get my workitem id
    int tid = get_global_id(0);

    if (tid < num_nodes){
        //get the starting and ending pointers of the neighborlist
        int start = row[tid];
        int end;
        if (tid + 1 < num_nodes)
           end = row[tid + 1] ;
        else
           end = num_edges;

        int nid;
        const float myPgRkVal = page_rank1[tid]/(float)(end-start);
		//navigate the neighbor list
        for(int edge = start; edge < end; edge++){
            nid = col[edge];
            //transfer the PageRank value to neighbors
            add_float_atomic(&page_rank2[nid], myPgRkVal);
        }
    }

}

/**
 * @brief   pagerank 2
 * @param   row         csr pointer array 
 * @param   col         csr column array 
 * @param   data        weight array 
 * @param   page_rank1  pagerank array 1
 * @param   page_rank2  pagerank array 2
 * @param   num_nodes   number of vertices
 * @param   num_edges   number of edges
 */
__kernel void pagerank2(__global int   *row,
                        __global int   *col,
                        __global int   *data,
                        __global float *page_rank1,
                        __global float *page_rank2,
                             const int num_nodes,
                             const int num_edges){

    //get my workitem id
    int tid = get_global_id(0);

    //update pagerank value with the damping factor
    if (tid < num_nodes){
        page_rank1[tid]	= 0.15/(float)num_nodes + 0.85 * page_rank2[tid];	
        page_rank2[tid] = 0.0f;
     }
}

/**
 * @brief   inibuffer
 * @param   row         csr pointer array
 * @param   page_rank1  pagerank array 1
 * @param   page_rank2  pagerank array 2
 * @param   num_nodes   number of vertices
 */
__kernel  void inibuffer( __global int   *row, 
                          __global float *page_rank1,
                          __global float *page_rank2,
                            const  int   num_nodes,
                            const  int   num_edges )
{
    //get my thread id
    int tid = get_global_id(0);

    if (tid < num_nodes){
       page_rank1[tid] = 1 / (float)num_nodes;
       page_rank2[tid] = 0.0f;
    }

}





