#include "kernel.h"
#include "cuda_func.cuh"

#define TX2 32
#define TY2 32
#define TX 8
#define TY 8
#define TZ 8
#define STD 0.01f

void importKernel(Npp8u *d_img, int4 imgSize, int4 volSize)
{
	Npp8u *h_in = (Npp8u*)malloc(imgSize.x*imgSize.y*imgSize.w*sizeof(Npp8u));
	Npp8u *h_xy = (Npp8u*)malloc(volSize.x*volSize.y*imgSize.z*volSize.w*sizeof(Npp8u));
	Npp8u *h_yz = (Npp8u*)malloc(volSize.y*imgSize.z*volSize.w*sizeof(Npp8u));
	Npp8u *h_xyz = (Npp8u*)malloc(volSize.y*volSize.z*volSize.w*sizeof(Npp8u));
	Npp8u *h_img = (Npp8u*)malloc(volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));

	Npp8u *d_in = 0, *d_xy = 0, *d_yz = 0, *d_xyz = 0;
	cudaMalloc(&d_in, imgSize.x*imgSize.y*imgSize.w*sizeof(Npp8u));
	cudaMalloc(&d_xy, volSize.x*volSize.y*imgSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_yz, volSize.y*imgSize.z*volSize.w*sizeof(Npp8u));
	cudaMalloc(&d_xyz, volSize.y*volSize.z*volSize.w*sizeof(Npp8u));

	for (int s = 0; s < imgSize.z; ++s)
	{
		char importFile[28];
		sprintf(importFile, "Color_arm_1/arm_%i.bmp", s + 1);
		cimg_library::CImg<unsigned char> imgIn(importFile);

		for (int r = 0; r < imgSize.y; ++r)
		{
			for (int c = 0; c < imgSize.x; ++c)
			{
				for (int ch = 0; ch < imgSize.w; ++ch)
				{
					int i = imgSize.w*imgSize.x*r + imgSize.w*c + ch;
					h_in[i] = imgIn(c, r, ch);
				}
			}
		}
		cudaMemcpy(d_in, h_in, imgSize.x*imgSize.y*imgSize.w*sizeof(Npp8u), cudaMemcpyHostToDevice);

		Npp8u *d_out = &d_xy[s*volSize.x*volSize.y*volSize.w];
		NppiSize oSrcSize = { imgSize.x, imgSize.y }, dstROISize = { volSize.x, volSize.y };
		int nSrcStep = imgSize.x*imgSize.w*sizeof(Npp8u), nDstStep = volSize.x*volSize.w*sizeof(Npp8u);
		NppiRect oSrcROI = { 0, 0, imgSize.x, imgSize.y };
		double nXFactor = (double)volSize.x / imgSize.x, nYFactor = (double)volSize.y / imgSize.y;

		nppiResize_8u_C3R(d_in, oSrcSize, nSrcStep, oSrcROI, d_out, nDstStep, dstROISize, nXFactor, nYFactor, 1);
	}
	cudaMemcpy(h_xy, d_xy, volSize.x*volSize.y*imgSize.z*volSize.w*sizeof(Npp8u), cudaMemcpyDeviceToHost);

	for (int c = 0; c < volSize.x; ++c)
	{
		for (int s = 0; s < imgSize.z; ++s)
		{
			for (int r = 0; r < volSize.y; ++r)
			{
				for (int ch = 0; ch < volSize.w; ++ch)
				{
					int i = volSize.w*volSize.y*volSize.x*s + volSize.w*volSize.x*r + volSize.w*c + ch;
					int j = volSize.w*volSize.y*s + volSize.w*r + ch;
					h_yz[j] = h_xy[i];
				}
			}
		}
		cudaMemcpy(d_yz, h_yz, volSize.y*imgSize.z*volSize.w*sizeof(Npp8u), cudaMemcpyHostToDevice);
		
		NppiSize oSrcSize = { volSize.y, imgSize.z }, dstROISize = { volSize.y, volSize.z };
		int nSrcStep = volSize.y*volSize.w*sizeof(Npp8u), nDstStep = volSize.y*volSize.w*sizeof(Npp8u);
		NppiRect oSrcROI = { 0, 0, volSize.y, imgSize.z };
		double nXFactor = (double)volSize.y / volSize.y, nYFactor = (double)volSize.z / imgSize.z;

		nppiResize_8u_C3R(d_yz, oSrcSize, nSrcStep, oSrcROI, d_xyz, nDstStep, dstROISize, nXFactor, nYFactor, 1);

		cudaMemcpy(h_xyz, d_xyz, volSize.y*volSize.z*volSize.w*sizeof(Npp8u), cudaMemcpyDeviceToHost);
		for (int s = 0; s < volSize.z; ++s)
		{
			for (int r = 0; r < volSize.y; ++r)
			{
				for (int ch = 0; ch < volSize.w; ++ch)
				{
					int i = volSize.w*volSize.y*volSize.x*s + volSize.w*volSize.x*r + volSize.w*c + ch;
					int j = volSize.w*volSize.y*s + volSize.w*r + ch;
					h_img[i] = h_xyz[j];
				}
			}
		}
	}

	cudaMemcpy(d_img, h_img, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u), cudaMemcpyHostToDevice);

	free(h_in);
	free(h_xy);
	free(h_yz);
	free(h_xyz);
	free(h_img);
	cudaFree(d_in);
	cudaFree(d_xy);
	cudaFree(d_yz);
	cudaFree(d_xyz);
}

void exportKernel(Npp8u *d_img, int4 volSize)
{
	Npp8u *img = (Npp8u*)malloc(volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u));;
	cudaMemcpy(img, d_img, volSize.x*volSize.y*volSize.z*volSize.w*sizeof(Npp8u), cudaMemcpyDeviceToHost);

	for (int s = 0; s < volSize.z; ++s)
	{
		char exportFile[28];
		sprintf(exportFile, "CUDA_Arm/arm_%i.bmp", s + 1);
		cimg_library::CImg<unsigned char> imgOut(volSize.x, volSize.y, 1, volSize.w);

		for (int r = 0; r < volSize.y; ++r)
		{
			for (int c = 0; c < volSize.x; ++c)
			{
				for (int ch = 0; ch < volSize.w; ++ch)
				{
					int i = volSize.w*(s*volSize.x*volSize.y + r*volSize.x + c) + ch;
					imgOut(c, r, ch) = img[i];
				}
			}
		}
		imgOut.save_bmp(exportFile);
	}

	free(img);
}