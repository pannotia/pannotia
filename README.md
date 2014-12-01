Pannotia v0.9
=============

© 2014 Advanced Micro Devices, Inc. All rights reserved.

Pannotia is a suite of parallel graph applications.
It is implemented with OpenCL and consists of applications from
diverse graph domains such as shortest path, graph partitioning, 
and web and graph analytics.

Here is the link for Pannotia wiki: https://github.com/pannotia/pannotia/wiki

System Requirements
===================

1. Linux® (tested with Ubuntu 12.04)
2. AMD APP SDK v2.8 and onwards
3. The latest AMD Catalyst™ driver
4. AMD APUs and GPUs with OpenCL™ 1.0 support

Installation, Building and Running
==================================

1. Check out the Pannotia repository: git clone https://github.com/pannotia/pannotia.git
2. Update the make.config file in the common folder to include the correct paths
3. Use the makefiles compile and build programs
4. Run the programs using the sample commands in the makefiles (e.g., make <run_input>)

Notices
=======

If your use of Pannotia results in a publication, please cite: 

* S. Che ( Advanced Micro Devices, Inc.), B. M. Beckmann ( Advanced Micro Devices, Inc.), S. K. Reinhardt ( Advanced Micro Devices, Inc.)  and K. Skadron ( University of Virginia), Pannotia: Understanding Irregular GPGPU Graph Applications. In Proceedings of the IEEE International Symposium on Workload Characterization (IISWC), Sept. 2013.

Pannotia includes the input files G3_circuit.graph and ecology1.graph from: 

* T. A. Davis and Y. Hu. The University of Florida Sparse Matrix Collection. ACM Transactions on Mathematical Software, 38(1), Nov 2011.

Pannotia includes the input file coAuthorsDBLP.graph from:

* D. A. Bader, H. Meyerhenke, P. Sanders, D. Wagner (eds.): Graph Partitioning and Graph Clustering. 10th DIMACS Implementation Challenge Workshop. Feb, 2012. Georgia Institute of Technology, Atlanta , GA. Contemporary Mathematics 588. American Mathematical Society and Center for Discrete Mathematics and Theoretical Computer Science, 2013.

DISCLAIMER
=========

The information contained herein is for informational purposes only, and is subject to change without notice. While every 
precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and 
typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro 
Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this 
document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or 
fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described 
herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. 
Terms and limitations applicable to the purchase or use of AMD’s products are as set forth in a signed agreement between the 
parties or in AMD's Standard Terms and Conditions of Sale.

AMD, the AMD Arrow logo, AMD Catalyst and combinations thereof are trademarks of Advanced Micro Devices, Inc. OpenCL  is a registered trademark of Apple Inc.  Linux is the registered trademark of Linus Torvalds in the U.S. and other countries. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. 
