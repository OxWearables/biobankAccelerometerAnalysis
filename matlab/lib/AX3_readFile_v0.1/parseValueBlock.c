/* 
 * Copyright (c) 2012, Newcastle University, UK.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met: 
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 *    this list of conditions and the following disclaimer in the documentation 
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE. 
 */

// Matlab Parser for .CWA files - block unpacking C function
// Nils Hammerla, 2012

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    unsigned int* encoded = (unsigned int*) mxGetPr(prhs[0]);
    int len = mxGetM(prhs[0]);
    int i;
    double* decoded;
    
	plhs[0] = mxCreateDoubleMatrix(3, len, mxREAL);
    
	decoded = mxGetPr(plhs[0]);
    
    for (i=0; i < len; i++) {
        decoded[(i*3)]   = (double)( (short)((unsigned short)0xffc0 & (unsigned short)(encoded[i] <<  6)) >> (6 - ((unsigned char)(encoded[i] >> 30))) );
        decoded[(i*3)+1] = (double)( (short)((unsigned short)0xffc0 & (unsigned short)(encoded[i] >>  4)) >> (6 - ((unsigned char)(encoded[i] >> 30))) );
        decoded[(i*3)+2] = (double)( (short)((unsigned short)0xffc0 & (unsigned short)(encoded[i] >> 14)) >> (6 - ((unsigned char)(encoded[i] >> 30))) );
    }

} 
