#include "kernel.h"
#include "cuda_func.cuh"

#define TX2 32
#define TY2 32
#define TX 8
#define TY 8
#define TZ 8
#define STD 0.01f

void importNPP(Npp8u *d_img, int4 imgSize, int4 volSize)
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

void exportNPP(Npp8u *d_img, int4 volSize)
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

void colorSeparateNPP(Npp8u *d_img, Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize)
{
	Npp32s nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	Npp8u *pDst1 = 0;
	cudaMalloc(&pDst1, volSize.x*volSize.y*volSize.w*sizeof(Npp8u));
	const Npp32f aTwistBtoW[3][4] = {
		{ 0, 0, 255, 0 },
		{ 0, 0, 255, 0 },
		{ 0, 0, 255, 0 } };
	const Npp32f aTwistRtoR[3][4] = {
		{ 255, 0, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 } };
	const Npp32f aTwistRtoB[3][4] = {
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 255, 0, 0, 0 } };
	const Npp32f aTwistRtoG[3][4] = {
		{ 0, 0, 0, 0 },
		{ 255, 0, 0, 0 },
		{ 0, 0, 0, 0 } };
	const Npp32f aTwistGtoR[3][4] = {
		{ 0, 255, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 } };

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pSrc = &d_img[s*volSize.x*volSize.y*volSize.w];
		Npp8u *pDst = &d_bone[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pSrc, nStep, pDst, nStep, oSizeROI, aTwistBtoW);

		pDst = &d_fat[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pSrc, nStep, pDst, nStep, oSizeROI, aTwistRtoB);

		pDst = &d_skin[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pSrc, nStep, pDst, nStep, oSizeROI, aTwistRtoG);

		pDst = &d_muscle[s*volSize.x*volSize.y*volSize.w];
		Npp8u *pSrc1 = &d_bone[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pSrc, nStep, pDst, nStep, oSizeROI, aTwistRtoR);
		nppiColorTwist32f_8u_C3R(pSrc, nStep, pDst1, nStep, oSizeROI, aTwistGtoR);
		nppiSub_8u_C3IRSfs(pDst1, nStep, pDst, nStep, oSizeROI, 0);
		nppiColorTwist32f_8u_C3R(pSrc1, nStep, pDst1, nStep, oSizeROI, aTwistRtoR);
		nppiAdd_8u_C3IRSfs(pDst1, nStep, pDst, nStep, oSizeROI, 0);
	}
	cudaFree(pDst1);
}

void imageAddNPP(Npp8u *d_img, Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize)
{
	int nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	const Npp32f aTwist[3][4] = {
		{ 255, 255, 255, 0 },
		{ 255, 255, 255, 0 },
		{ 255, 255, 255, 0 } };

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pDst = &d_img[s*volSize.x*volSize.y*volSize.w];
		Npp8u *pSrc = &d_bone[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pSrc, nStep, pDst, nStep, oSizeROI, aTwist);
		pSrc = &d_muscle[s*volSize.x*volSize.y*volSize.w];
		nppiAdd_8u_C3IRSfs(pSrc, nStep, pDst, nStep, oSizeROI, 0);
		pSrc = &d_fat[s*volSize.x*volSize.y*volSize.w];
		nppiAdd_8u_C3IRSfs(pSrc, nStep, pDst, nStep, oSizeROI, 0);
		pSrc = &d_skin[s*volSize.x*volSize.y*volSize.w];
		nppiAdd_8u_C3IRSfs(pSrc, nStep, pDst, nStep, oSizeROI, 0);
	}
}

void boneNPP(Npp8u *d_bone, int *boneDandE, int4 volSize)
{
	Npp32s nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	NppiSize oMaskSize = { 3, 3 };
	NppiPoint oAnchor = { 1, 1 };
	const Npp8u h_mask[27] = {
		0, 1, 0, 1, 1, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 1, 0, 1, 0 };
	Npp8u *pMask = 0;
	cudaMalloc(&pMask, sizeof(h_mask));
	cudaMemcpy(pMask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice);

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pSrc = &d_bone[s*volSize.x*volSize.y*volSize.w];

		for (int i = 0; i < boneDandE[0]; ++i)
		{
			nppiDilate_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
		for (int i = 0; i < boneDandE[0] + boneDandE[1]; ++i)
		{
			nppiErode_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
		for (int i = 0; i < boneDandE[1] + boneDandE[2]; ++i)
		{
			nppiDilate_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
		for (int i = 0; i < boneDandE[2]; ++i)
		{
			nppiErode_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
	}
	cudaFree(pMask);
}

void muscleNPP(Npp8u *d_muscle, int *muscleDandE, int4 volSize)
{
	Npp32s nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	NppiSize oMaskSize = { 3, 3 };
	NppiPoint oAnchor = { 1, 1 };
	const Npp8u h_mask[27] = { 
		0, 1, 0, 1, 1, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 1, 0, 1, 0, 
		0, 1, 0, 1, 1, 1, 0, 1, 0 };
	Npp8u *pMask = 0;
	cudaMalloc(&pMask, sizeof(h_mask));
	cudaMemcpy(pMask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice);

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pSrc = &d_muscle[s*volSize.x*volSize.y*volSize.w];

		for (int i = 0; i < muscleDandE[0]; ++i)
		{
			nppiDilate_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
		for (int i = 0; i < muscleDandE[0] + muscleDandE[1]; ++i)
		{
			nppiErode_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
		for (int i = 0; i < muscleDandE[1] + muscleDandE[2]; ++i)
		{
			nppiDilate_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
		for (int i = 0; i < muscleDandE[2]; ++i)
		{
			nppiErode_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
	}
	cudaFree(pMask);
}

void fatNPP(Npp8u *d_fat, int blendDist, int4 volSize)
{
	Npp32s nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	NppiSize oKernelSize = { 3, 3 };
	NppiPoint oAnchor = { 1, 1 };
	const Npp32s h_Kernel[9] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
	const Npp32s nDivisor = 5;
	Npp32s *pKernel = 0;
	cudaMalloc(&pKernel, sizeof(h_Kernel));
	cudaMemcpy(pKernel, h_Kernel, sizeof(h_Kernel), cudaMemcpyHostToDevice);

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pSrc = &d_fat[s*volSize.x*volSize.y*volSize.w];
		for (int i = 0; i < blendDist; ++i)
		{
			nppiFilter_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor);
		}
	}
	cudaFree(pKernel);
}

void skinNPP(Npp8u *d_skin, Npp8u *d_fat, int skinThickness, int4 volSize)
{
	Npp32s nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	NppiSize oMaskSize = { 3, 3 };
	NppiPoint oAnchor = { 1, 1 };
	const Npp8u h_mask[27] = {
		0, 1, 0, 1, 1, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 1, 0, 1, 0 };
	Npp8u *pMask = 0;
	cudaMalloc(&pMask, sizeof(h_mask));
	cudaMemcpy(pMask, h_mask, sizeof(h_mask), cudaMemcpyHostToDevice);

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pSrc = &d_skin[s*volSize.x*volSize.y*volSize.w];
		
		for (int i = 0; i < skinThickness; ++i)
		{
			nppiDilate_8u_C3R(pSrc, nStep, pSrc, nStep, oSizeROI, pMask, oMaskSize, oAnchor);
		}
	}
	cudaFree(pMask);
}

void trimNPP(Npp8u *d_bone, Npp8u *d_muscle, Npp8u *d_fat, Npp8u *d_skin, int4 volSize)
{
	Npp32s nStep = volSize.x*volSize.w*sizeof(Npp8u);
	NppiSize oSizeROI = { volSize.x, volSize.y };
	Npp8u *pDst2 = 0;
	cudaMalloc(&pDst2, volSize.x*volSize.y*volSize.w*sizeof(Npp8u));
	const Npp32f aTwistWtoR[3][4] = {
		{ 1, 1, 1, 0 },
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 } };
	const Npp32f aTwistRtoB[3][4] = {
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 0 },
		{ 1, 0, 0, 0 } };
	const Npp32f aTwistBtoG[3][4] = {
		{ 0, 0, 0, 0 },
		{ 0, 0, 1, 0 },
		{ 0, 0, 0, 0 } };
	const Npp32f aTwistRtoG[3][4] = {
		{ 0, 0, 0, 0 },
		{ 1, 0, 0, 0 },
		{ 0, 0, 0, 0 } };

	for (int s = 0; s < volSize.z; ++s)
	{
		Npp8u *pSrc1 = &d_skin[s*volSize.x*volSize.y*volSize.w];
		Npp8u *pScr2 = &d_fat[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pScr2, nStep, pDst2, nStep, oSizeROI, aTwistBtoG);
		nppiSub_8u_C3IRSfs(pDst2, nStep, pSrc1, nStep, oSizeROI, 0);
		pScr2 = &d_muscle[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pScr2, nStep, pDst2, nStep, oSizeROI, aTwistRtoG);
		nppiSub_8u_C3IRSfs(pDst2, nStep, pSrc1, nStep, oSizeROI, 0);

		pSrc1 = &d_fat[s*volSize.x*volSize.y*volSize.w];
		pScr2 = &d_muscle[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pScr2, nStep, pDst2, nStep, oSizeROI, aTwistRtoB);
		nppiSub_8u_C3IRSfs(pDst2, nStep, pSrc1, nStep, oSizeROI, 0);

		pSrc1 = &d_muscle[s*volSize.x*volSize.y*volSize.w];
		pScr2 = &d_bone[s*volSize.x*volSize.y*volSize.w];
		nppiColorTwist32f_8u_C3R(pScr2, nStep, pDst2, nStep, oSizeROI, aTwistWtoR);
		nppiSub_8u_C3IRSfs(pDst2, nStep, pSrc1, nStep, oSizeROI, 0);
	}
	cudaFree(pDst2);
}