#include "kernel.h"

void importNPP(Npp16u *d_in, int3 imgSize)
{
	printf("Reading images...\n");
	Npp16u *h_in = (Npp16u*)malloc(imgSize.x*imgSize.y*imgSize.z*sizeof(Npp16u));
	
	for (int s = 0; s < imgSize.z; ++s)
	{
		char fileName[126];
		sprintf(fileName, "Easy/IM-0001-%04i.dcm", s + 1);
		//cimg_library::CImg<unsigned char>load_medcon_external(fileName);
		cimg_library::CImg<unsigned char>load_medcon_externel(fileName);
		for (int r = 0; r < imgSize.y; ++r)
		{
			for (int c = 0; c < imgSize.x; ++c)
			{
				int i = s*imgSize.y*imgSize.x + r*imgSize.x + c;
				//h_in[i] = imgIn(c, r);
			}
		}
	}
	cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
	
	free(h_in);
}