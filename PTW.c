#include "mex.h"
#include <math.h>
#include <stdlib.h>

void printf_complex_vec(double *real, double *image, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f + i %f\n",real[i], image[i]);
         mexEvalString("drawnow");
     }    
 }

void printf_vec(double *arr_double_ind, int n)
 {
     int i;
     for (i=0;i <n; ++i)
     {
         mexPrintf("%f\n",arr_double_ind[i]);
         mexEvalString("drawnow");
     }    
 }

double sum_c(double *arr, int n_row, int b)
{
    int j;
    double sum_=0;

    for (j=0; j<n_row; j++)
    {       
        sum_=sum_+arr[j+b*n_row];
    }
    return sum_;
}

//Note that in C, index= no_of_rows * col_id + row_id
double * project_row(double *arr, double *result, int row, int b, int no_of_rows, int no_of_cols)
{
    int i;
    
    for (i=0; i<no_of_cols; i++)
    {
        result[i]=arr[row+i*no_of_rows+b*(no_of_rows*no_of_cols)];
    }
    return result;
}


double * project_row_seg(double *arr, double *result, int m, int b)
{
    int i;
    for (i=0; i<m; i++)
    {
        result[i]=arr[i+b*m];
    }
    return result;
}

double * xcorr_valid(double *arr1, int size1, double *arr2, int size2, double *result, int width)
{
//     double *arr1_new = (double*)mxCalloc((mwSize)(size1+width-1),(mwSize)sizeof(double));
//     double *result= (double*)mxCalloc((mwSize)width,(mwSize)sizeof(double));
    int n, k, t;
//     if (width%2==0)
//     {
//         t=width/2-1;
//     } else
//     {
//         t=width/2;
//     }
//     for (n=0; n<size1+width-1; n++)
//     {
//         if (n < t || n > size1+t-1)
//         {
//             if (n < t) arr1_new[n]=arr1[0];
//             if (n > size1+t-1) arr1_new[n]=arr1[size1-1];
// //             arr1_new[n]=0;
//         } else
//         {
//             arr1_new[n]=arr1[n-t];
//         }
//     }
// //     printf_vec(arr1_new, size1+width-1);
    for (n=0; n<width; n++)
    {
        for ( k = 0; k < size2; k++)
        {
//            result[n]+=arr1_new[n+k]*arr2[k];
            result[n]+=arr1[n+k]*arr2[k];
        }
    }
//     mxFree(arr1_new);
    return result;
}

void FFT(short int dir,long m,double *x,double *y)
{
   long n,i,i1,j,k,i2,l,l1,l2;
   double c1,c2,tx,ty,t1,t2,u1,u2,z;

   /* Calculate the number of points */
   n = 1;
   for (i=0;i<m;i++) 
      n *= 2;

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;
   for (i=0;i<n-1;i++) {
      if (i < j) {
         tx = x[i];
         ty = y[i];
         x[i] = x[j];
         y[i] = y[j];
         x[j] = tx;
         y[j] = ty;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }

   /* Compute the FFT */
   c1 = -1.0; 
   c2 = 0.0;
   l2 = 1;
   for (l=0;l<m;l++) {
      l1 = l2;
      l2 <<= 1;
      u1 = 1.0; 
      u2 = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<n;i+=l2) {
            i1 = i + l1;
            t1 = u1 * x[i1] - u2 * y[i1];
            t2 = u1 * y[i1] + u2 * x[i1];
            x[i1] = x[i] - t1; 
            y[i1] = y[i] - t2;
            x[i] += t1;
            y[i] += t2;
         }
         z =  u1 * c1 - u2 * c2;
         u2 = u1 * c2 + u2 * c1;
         u1 = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (dir == 1) 
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
   }

   /* Scaling for backward transform */
   if (dir == -1) {
      for (i=0;i<n;i++) {
         x[i] /= n;
         y[i] /= n;
      }
   }
   
}


void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *arr1;
    double *arr2;
    double *sig1, *sig2, *tempsig, *result;
    double *fftrealsig1, *fftrealsig2, *ifftrealsig3;
    double *fftimgsig1, *fftimgsig2, *ifftimgsig3;
    int i,t, b,f,length;
    int bool_addone, width;
    int bool_conv;
    int NFFT;
    int nDimNumP, nDimNumX;
    const int *pDimsP;
    const int *pDimsX;
    int nDimNum1;
    int *pDims1;
    mxArray *Data;
    size_t size1;     
    size_t size2;
    size_t F,B;
    double *outarray;  
  
    arr1 = mxGetPr(prhs[0]);
    arr2 = mxGetPr(prhs[1]); 
    width = mxGetScalar(prhs[2]);
    //zero padding
//     if (width%2==0)
//     {
//         t=width/2-1;
//     } else
//     {
//         t=width/2;
//     }

    bool_addone = mxGetScalar(prhs[3]);
    bool_conv = mxGetScalar(prhs[4]);
//     mexPrintf("%d, %d\n", width, bool_addone);
    nDimNumX = mxGetNumberOfDimensions(prhs[0]);
    pDimsX = mxGetDimensions(prhs[0]);
    F = pDimsX[0];
    size1 = pDimsX[1];
    B = (nDimNumX==2)? 1: pDimsX[2];
    
    nDimNumP = mxGetNumberOfDimensions(prhs[1]);
//     mexPrintf("%d, %d, %d\n", F, size1, nDimNum);
    pDimsP = mxGetDimensions(prhs[1]);
    size2=pDimsP[0];
    NFFT = (int)pow(2.0, ceil(log((double)size1+size2-1)/log(2.0)));
       
    nDimNum1 = (nDimNumX==2)? nDimNumX: nDimNumX-1;
    pDims1 = (int*)mxCalloc((mwSize) nDimNum1,(mwSize)sizeof(int));
    if (bool_addone==1)
    {    
        pDims1[0]=width*F+1;
    }else{
        pDims1[0]=width*F;    
    }
    pDims1[1]=B;
//     mexPrintf("%d, %d, %d, %d\n", size2, pDims2[0], C, K);
    Data = mxCreateNumericArray(nDimNum1, pDims1, mxDOUBLE_CLASS, mxREAL);
    result = (double *) mxGetPr(Data);

    for (b=0; b<B; b++) {
        for (i=0; i<pDims1[0]; i++){
            result[i+b*pDims1[0]]=0;
        }
    }
    
    sig1 = (double*)mxCalloc((mwSize)size1,(mwSize)sizeof(double));
    sig2 = (double*)mxCalloc((mwSize)size2,(mwSize)sizeof(double));
    tempsig = (double*)mxCalloc((mwSize)width,(mwSize)sizeof(double));
    fftrealsig1 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    fftimgsig1 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    fftrealsig2 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    fftimgsig2 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    ifftrealsig3 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    ifftimgsig3 = (double*)mxCalloc((mwSize)NFFT,(mwSize)sizeof(double));
    
    for (b=0; b<B; b++){
        for (f=0; f<F; f++){
            for (i=0; i<size1; i++){
                sig1[i]=0;
            }
            sig1 = project_row(arr1, sig1, f, b, F, size1);
            for (i=0; i<size2; i++){
                sig2[i]=0;
            }
            sig2 = project_row_seg(arr2, sig2, size2, b);
            if (bool_conv) {
                for (i=0; i<width; i++){
                    tempsig[i] = 0;
                }
                tempsig = xcorr_valid(sig2, size2, sig1, size1, tempsig, width);
            } else {

                for (i=0; i<NFFT; i++) {
                    fftrealsig1[i]=0;
                    fftimgsig1[i]=0;
                    fftrealsig2[i]=0;
                    fftimgsig2[i]=0;
                }

//                 for (i=0; i<size1+size2-1; i++) {
//                     if (i < t|| i > size1+t-1)
//                     {
//                         if (i < t) fftrealsig1[i]=sig1[0];
//                         if (i > size1+t-1) fftrealsig1[i]=sig1[size1-1];
//                     } else
//                         fftrealsig1[i]=sig1[i-t];
//                 }
                
                for (i=0; i<size1; i++) {
//                     fftrealsig1[i]=sig1[i];
                    fftrealsig1[i]=sig1[size1-1-i];
                }

                FFT(1,(int)(log((double)NFFT)/log(2.0)),fftrealsig1, fftimgsig1);

                for (i=0; i<size2; i++) {
                    fftrealsig2[i]=sig2[i];
//                     fftrealsig2[i]=sig2[size2-1-i];
                }

                FFT(1,(int)(log((double)NFFT)/log(2.0)),fftrealsig2, fftimgsig2);

                for (i=0; i<NFFT; i++) {
//                                 mexPrintf("fftrsig1: %f, fftrsig2: %f for h=%d",fftrealsig1[h], fftrealsig2[h],h);
                    ifftrealsig3[i] = fftrealsig1[i] * fftrealsig2[i] - fftimgsig1[i] * fftimgsig2[i];
//                                 mexPrintf("fftisig1: %f, fftisig2: %f for h=%d",fftimgsig1[h], fftimgsig2[h],h);
                    ifftimgsig3[i] = fftrealsig1[i] * fftimgsig2[i] + fftrealsig2[i]* fftimgsig1[i] ;
                }
                FFT(-1,(int)(log((double)NFFT)/log(2.0)),ifftrealsig3, ifftimgsig3);
//                         if (f==0 && b==0) {
//                             mexPrintf("fft of signal 1 for b= %d, f= %d\n", b, f);
//                             printf_complex_vec(fftrealsig1, fftimgsig1, NFFT);
//                             mexPrintf("fft of signal 2 for b= %d, f= %d\n", b, f);
//                             printf_complex_vec(fftrealsig2, fftimgsig2, NFFT);
//                             mexPrintf("ifft of signal 3 for b= %d, f= %d\n", b, f);
//                             printf_complex_vec(ifftrealsig3, ifftimgsig3, NFFT);
//                         }
                for (i=0; i<width; i++)
                {
                    tempsig[i] = ifftrealsig3[i+size1-1];
                }
            }
//                 if (f<=4)
//                 {
//                     mexPrintf("%d\n", f);
//                     printf_vec(sig1,size1);
//                     printf_vec(sig2,size2);
//                     printf_vec(tempsig,width);
//                 }
            for (i=0; i<width; i++){
                result[f*width+i+b*pDims1[0]] = tempsig[i];
            }
        }
        if(bool_addone==1)
        {      
            result[pDims1[0]-1+b*pDims1[0]]=sum_c(arr2, size2, b);
    //                 mexPrintf("%f\n", result[pDims2[0]-1+c*pDims2[0]+k*pDims2[0]*C]);
        }

    }

    mxFree(sig1);
    mxFree(sig2);
    mxFree(tempsig);
    mxFree(fftrealsig1);
    mxFree(fftrealsig2);
    mxFree(ifftrealsig3);
    mxFree(fftimgsig1);
    mxFree(fftimgsig2);
    mxFree(ifftimgsig3);
    
    length = (int) pDims1[0]*B;
    plhs[0] = mxCreateNumericArray(nDimNum1, pDims1, mxDOUBLE_CLASS, mxREAL);
    outarray = (double *)mxGetPr(plhs[0]);
    memcpy(outarray,result, length*sizeof(double));
    }
    
     
    
