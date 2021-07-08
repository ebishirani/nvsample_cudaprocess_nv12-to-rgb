/*
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include<iostream>
#include<fstream>

#include <cuda.h>

// #include <cudaYUV.h>
#include <kcftracker.hpp>

#include "customer_functions.h"
#include "cudaEGL.h"
#include "iva_metadata.h"

#define BOX_W 32
#define BOX_H 32

#define CORD_X 64
#define CORD_Y 64
#define MAX_BUFFERS 30
static BBOX rect_data[MAX_BUFFERS];

#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define FIXED_DECIMAL_POINT             24
#define FIXED_POINT_MULTIPLIER          1.0f
#define FIXED_COLOR_COMPONENT_MASK      0xffffffff

//-----------------------------------------------------------------------------------
// YUV to RGB colorspace conversion
//-----------------------------------------------------------------------------------
// static compile-time assertion
template<typename T> struct cuda_assert_false : std::false_type { };
// get base type (uint8 or float) from vector
template<class T> struct cudaVectorTypeInfo;

template<> struct cudaVectorTypeInfo<uchar>  { typedef uint8_t Base; };
template<> struct cudaVectorTypeInfo<uchar3> { typedef uint8_t Base; };
template<> struct cudaVectorTypeInfo<uchar4> { typedef uint8_t Base; };

template<> struct cudaVectorTypeInfo<float>  { typedef float Base; };
template<> struct cudaVectorTypeInfo<float3> { typedef float Base; };
template<> struct cudaVectorTypeInfo<float4> { typedef float Base; };
// make_vec<T> templates
template<typename T> inline __host__ __device__ T make_vec(typename cudaVectorTypeInfo<T>::Base x, typename cudaVectorTypeInfo<T>::Base y, typename cudaVectorTypeInfo<T>::Base z, typename cudaVectorTypeInfo<T>::Base w )	{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }

template<> inline __host__ __device__ uchar  make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return x; }
template<> inline __host__ __device__ uchar3 make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return make_uchar3(x,y,z); }
template<> inline __host__ __device__ uchar4 make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return make_uchar4(x,y,z,w); }

template<> inline __host__ __device__ float  make_vec( float x, float y, float z, float w )		{ return x; }
template<> inline __host__ __device__ float3 make_vec( float x, float y, float z, float w )		{ return make_float3(x,y,z); }
template<> inline __host__ __device__ float4 make_vec( float x, float y, float z, float w )		{ return make_float4(x,y,z,w); }

static inline __device__ float clamp( float x )	{ return fminf(fmaxf(x, 0.0f), 255.0f); }

// YUV2RGB
template<typename T>
static inline __device__ T YUV2RGB(const uint3& yuvi)
{
	const float luma = float(yuvi.x);
	const float u    = float(yuvi.y) - 512.0f;
	const float v    = float(yuvi.z) - 512.0f;
	const float s    = 1.0f / 1024.0f * 255.0f;	// TODO clamp for uchar output?

#if 1
	return make_vec<T>(clamp((luma + 1.402f * v) * s),
				    clamp((luma - 0.344f * u - 0.714f * v) * s),
				    clamp((luma + 1.772f * u) * s), 255);
#else
	return make_vec<T>(clamp((luma + 1.140f * v) * s),
				    clamp((luma - 0.395f * u - 0.581f * v) * s),
				    clamp((luma + 2.032f * u) * s), 255);
#endif
}
//****************************************************************************
template<typename T>
__global__ void NV12ToRGB(
	void* yPlane, void* uvPlane, 
	size_t nSourcePitch,
	T* dstImage, size_t nDestPitch,
	uint32_t width, uint32_t height,
	int topX = 0, int topY = 0,
  int buttomX = 0, int buttomY = 0)
{
	int x, y;
	uint32_t yuv101010Pel[2];
	uint8_t *srcYPlane = (uint8_t *)yPlane;
	uint8_t *srcUVPlane = (uint8_t *)uvPlane;

	uint32_t processingPitch = nSourcePitch;

	// Pad borders with duplicate pixels, and we multiply by 2 because
	// we process 2 pixels per thread
	x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
	y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width )
  {
		return; //x = width - 1;
  }
	if( y >= height )
  {
		return; // y = height - 1;
  }
  if(((buttomX - topX) > 0) && ((buttomY - topY) > 0))
  {
      if(x < topX || x > buttomX)
      {
        return;
      }
      if(y < topY || y > buttomY)
      {
        return;
      }
  }

  // if(x == topX + 10 && y == topY + 10)
  // {
  //   printf("topx = %d\ttopY=%d\tbuttomX=%d\tbuttomY=%d\n", topX, topY,
  //           buttomX, buttomY);
  // }

	// Read 2 Luma components at a time, so we don't waste processing 
	//since CbCr are decimated this way.
	// if we move to texture we could read 4 luminance values
	yuv101010Pel[0] = (srcYPlane[y * processingPitch + x]) << 2;
	yuv101010Pel[1] = (srcYPlane[y * processingPitch + x + 1]) << 2;
	int y_chroma = y >> 1;

	if (y & 1)  // odd scanline ?
	{
		uint32_t chromaCb;
		uint32_t chromaCr;

		chromaCb = srcUVPlane[y_chroma * processingPitch + x];
		chromaCr = srcUVPlane[y_chroma * processingPitch + x + 1];
		if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
		{
			chromaCb = (chromaCb + 
				srcUVPlane[(y_chroma + 1) * processingPitch + x] + 1) >> 1;

			chromaCr = (chromaCr + 
				srcUVPlane[(y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
		}
		yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
		yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}
	else
	{
		yuv101010Pel[0] |= ((uint32_t)srcUVPlane[y_chroma * processingPitch + x] << 
		                   (COLOR_COMPONENT_BIT_SIZE + 2));

		yuv101010Pel[0] |= ((uint32_t)srcUVPlane[y_chroma * processingPitch + x + 1] <<
		                   ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
		yuv101010Pel[1] |= ((uint32_t)srcUVPlane[y_chroma * processingPitch + x] << 
		                   (COLOR_COMPONENT_BIT_SIZE + 2));
		yuv101010Pel[1] |= ((uint32_t)srcUVPlane[y_chroma * processingPitch + x + 1] <<
		                   ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
	}
	// this steps performs the color conversion
	const uint3 yuvi_0 = make_uint3((yuv101010Pel[0] &   COLOR_COMPONENT_MASK),
	                               ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK),
					               ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK));
	const uint3 yuvi_1 = make_uint3((yuv101010Pel[1] &   COLOR_COMPONENT_MASK),
							       ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK),
								   ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK));
	// YUV to RGB transformation conversion
  int yDest = y - topY;
  int xDest = x - topX;
  int wDest = buttomX - topX;
	dstImage[yDest * wDest + xDest] = YUV2RGB<T>(yuvi_0);
	dstImage[yDest * wDest + xDest + 1] = YUV2RGB<T>(yuvi_1);
}
//****************************************************************************
/**
  * Dummy custom pre-process API implematation.
  * It just access mapped surface userspace pointer &
  * memset with specific pattern modifying pixel-data in-place.
  *
  * @param sBaseAddr  : Mapped Surfaces pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param nsurfcount : surfaces count
  */
static void
pre_process (void **sBaseAddr,
                unsigned int *smemsize,
                unsigned int *swidth,
                unsigned int *sheight,
                unsigned int *spitch,
                ColorFormat  *sformat,
                unsigned int nsurfcount,
                void ** usrptr)
{
  /* add your custom pre-process here
     we draw a green block for demo */
  int x, y;
  char * uv = NULL;
  unsigned char * rgba = NULL;
  if (sformat[1] == COLOR_FORMAT_U8_V8) {
    uv = (char *)sBaseAddr[1];
    for (y = 0; y < BOX_H; ++y) {
      for (x = 0; x < BOX_W; ++x) {
        uv[y * spitch[1] + 2 * x] = 255;
        uv[y * spitch[1] + 2 * x + 1] = 255;
      }
    }
  } else if (sformat[0] == COLOR_FORMAT_RGBA) {
    rgba = (unsigned char *)sBaseAddr[0];
     for (y = 0; y < BOX_H*2; y++) {
      for (x = 0; x < BOX_W*8; x+=4) {
       rgba[x + 0] = 0;
       rgba[x + 1] = 0;
       rgba[x + 2] = 0;
       rgba[x + 3] = 0;
      }
        rgba+=spitch[0];
    }
  }
}
//****************************************************************************
/**
  * Dummy custom post-process API implematation.
  * It just access mapped surface userspace pointer &
  * memset with specific pattern modifying pixel-data in-place.
  *
  * @param sBaseAddr  : Mapped Surfaces pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param nsurfcount : surfaces count
  */
static void
post_process (void **sBaseAddr,
                unsigned int *smemsize,
                unsigned int *swidth,
                unsigned int *sheight,
                unsigned int *spitch,
                ColorFormat  *sformat,
                unsigned int nsurfcount,
                void ** usrptr)
{
  /* add your custom post-process here
     we draw a green block for demo */
  int x, y;
  char * uv = NULL;
  int xoffset = (CORD_X * 4);
  int yoffset = (CORD_Y * 2);
  unsigned char * rgba = NULL;
  if (sformat[1] == COLOR_FORMAT_U8_V8) {
    uv = (char *)sBaseAddr[1];
    for (y = 0; y < BOX_H; ++y) {
      for (x = 0; x < BOX_W; ++x) {
        uv[(y + BOX_H * 2) * spitch[1] + 2 * (x + BOX_W * 2)] = 255;
        uv[(y + BOX_H * 2) * spitch[1] + 2 * (x + BOX_W * 2) + 1] = 255;
      }
    }
  } else if (sformat[0] == COLOR_FORMAT_RGBA) {
    rgba = (unsigned char *)sBaseAddr[0];
    rgba += ((spitch[0] * yoffset) + xoffset);
     for (y = 0; y < BOX_H*2; y++) {
      for (x = 0; x < BOX_W*8; x+=4) {
       rgba[(x + xoffset) + 0] = 0;
       rgba[(x + xoffset) + 1] = 0;
       rgba[(x + xoffset) + 2] = 0;
       rgba[(x + xoffset) + 3] = 0;
      }
        rgba+=spitch[0];
    }
  }
}
//****************************************************************************
__global__ void addLabelsKernel(int* pDevPtr, int pitch){
  int row = blockIdx.y*blockDim.y + threadIdx.y + BOX_H;
  int col = blockIdx.x*blockDim.x + threadIdx.x + BOX_W;
  char * pElement = (char*)pDevPtr + row * pitch + col * 2;
  pElement[0] = 0;
  pElement[1] = 0;
  return;
}
//****************************************************************************
static int addLabels(CUdeviceptr pDevPtr, int pitch){
    dim3 threadsPerBlock(BOX_W, BOX_H);
    dim3 blocks(1,1);
    addLabelsKernel<<<blocks,threadsPerBlock>>>((int*)pDevPtr, pitch);
    return 0;
}
//****************************************************************************
static void add_metadata(void ** usrptr)
{
    /* User need to fill rectangle data based on their requirement.
     * Here rectangle data is filled for demonstration purpose only */

    int i;
    static int index = 0;

    rect_data[index].framecnt = index;
    rect_data[index].objectcnt = index;

    for(i=0; i < NUM_LOCATIONS; i++)
    {
        rect_data[index].location_list[i].x1 = index;
        rect_data[index].location_list[i].x2 = index;
        rect_data[index].location_list[i].y1 = index;
        rect_data[index].location_list[i].y2 = index;
    }
    *usrptr = &rect_data[index];
    index++;
    if(!(index % MAX_BUFFERS))
    {
        index = 0;
    }
}
//****************************************************************************
/**
  * Performs CUDA Operations on egl image.
  *
  * @param image : EGL image
  */
static void
gpu_process (EGLImageKHR image, void ** usrptr)
{
  CUresult status;
  CUeglFrame eglFrame;
  CUgraphicsResource pResource = NULL;

  cudaFree(0);
  status = cuGraphicsEGLRegisterImage(
    &pResource, 
    image, 
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);

  if (status != CUDA_SUCCESS) 
  {
    printf("cuGraphicsEGLRegisterImage failed : %d \n", status);
    return;
  }

  status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
  if (status != CUDA_SUCCESS)
  {
    printf ("cuGraphicsSubResourceGetMappedArray failed\n");
  }

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS)
  {
    printf ("cuCtxSynchronize failed \n");
  }

  int4 roi = {300, 100, 800, 600};
  int roiWidth = roi.z - roi.x;
  int roiHeight = roi.w - roi.y;
  size_t cropSize = roiWidth * roiHeight * 3 * sizeof(uchar);
  size_t cropPitch = roiWidth * 3 * sizeof(uchar);

  uchar3 *devRgbCropedImage = nullptr;
  uchar3 hostRgbCropedImage[cropSize];

  cudaMalloc(&devRgbCropedImage, cropSize);

  if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH)
  {
    if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_ABGR)
    {
      // printf("Salam dakhele cuda process hASTIM_____0\n");
      /* Rectangle label in plane RGBA, you can replace this with any cuda algorithms */
      addLabels((CUdeviceptr) eglFrame.frame.pPitch[0], eglFrame.pitch);
      // printf("Salam dakhele cuda process hASTIM_____0-1\n");
    } 
    else if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR)
    {
      // printf("Salam dakhele cuda process hASTIM_____1\n");
      /* Rectangle label in plan UV , you can replace this with any cuda algorithms */
      addLabels((CUdeviceptr) eglFrame.frame.pPitch[1], eglFrame.pitch);
      // printf("Salam dakhele cuda process hASTIM_____2\n");
      // printf("roi location is x = %d\ty = %d\tz = %d\tw = %d\n",
              // roi.x, roi.y, roi.z, roi.w);
      // printf("Salam dakhele cuda process hASTIM_____3\n");

      const dim3 blockDim(32,8,1);

      int widthDiv = (int)ceil(((float)eglFrame.width) / blockDim.x);
      int heightDiv = (int)ceil(((float)eglFrame.height) / blockDim.y);
	    const dim3 gridDim(widthDiv, heightDiv, 1);

      NV12ToRGB<uchar3><<<gridDim, blockDim>>>(
                              (void *)(eglFrame.frame.pPitch[0]),
                              (void *)(eglFrame.frame.pPitch[1]), 
                              eglFrame.pitch, devRgbCropedImage,
                              cropPitch, eglFrame.width, eglFrame.height,
                              roi.x, roi.y, roi.z, roi.w);
      // printf("Salam dakhele cuda process hASTIM_____4\n");
      // printf("Salam dakhele cuda process hASTIM_____5\n");
      
    }
    else
    {
      printf ("Invalid eglcolorformat\n");
    }
  }

  // printf("Salam dakhele cuda process hASTIM_____6\n");
  add_metadata(usrptr);
  // printf("Salam dakhele cuda process hASTIM_____7\n");

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS)
  {
    printf ("cuCtxSynchronize failed after memcpy \n");
  }
  // printf("Salam dakhele cuda process hASTIM_____8\n");
  cudaMemcpy(hostRgbCropedImage, devRgbCropedImage,
             cropSize, cudaMemcpyDeviceToHost);

  static int count = 0;
  if (status != CUDA_SUCCESS)
  {
    printf ("cuCtxSynchronize failed after memcpy \n");
  }
  else
  {    
    if(count == 0)
    {      
      count = 1;
      ofstream wf("rgbImage.dat", ios::out | ios::binary);
      wf.write((char *)hostRgbCropedImage, cropSize);
      wf.close();
    }
  }
  cudaFree(devRgbCropedImage);
  // printf("Salam dakhele cuda process hASTIM_____9\n");
  status = cuGraphicsUnregisterResource(pResource);
  if (status != CUDA_SUCCESS) {
    // printf("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
  }
  // printf("Salam dakhele cuda process hASTIM_____10\n");
}

extern "C" void
init (CustomerFunction * pFuncs)
{
  pFuncs->fPreProcess = pre_process;
  pFuncs->fGPUProcess = gpu_process;
  pFuncs->fPostProcess = post_process;
}

extern "C" void
deinit (void)
{
  /* deinitialization */
}
