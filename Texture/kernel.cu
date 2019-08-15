#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stdio.h>
using namespace std;
using namespace cv;

// cudaReadModeElementType不支持线性插值
// cudaReadModeNormalizedFloat支持线性插值
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex;


__global__ void smooth_kernel(char *img, int width, int heigth, int channels)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	//若使用归一化
	float u = x / (float)width;
	float v = y / (float)heigth;

	//如果使用cudaReadModeElementType，则读取uchar4不能转为float
	float4 pixel = tex2D(tex, u, v);
	float4 left = tex2D(tex, u - 1, v);
	float4 right = tex2D(tex, u + 1, v);
	float4 top = tex2D(tex, u, v - 1);
	float4 botton = tex2D(tex, u, v + 1);

	char* pix = img + (v*width + u)*channels;
	pix[0] = (left.x + right.x + top.x + botton.x) / 4 * 255;
	pix[1] = (left.y + right.y + top.y + botton.y) / 4 * 255;
	pix[2] = (left.z + right.z + top.z + botton.z) / 4 * 255;
	pix[3] = 0;
}

__global__ void zoom_kernel(char *img, int width, int heigth, int channels)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// 使用归一化坐标，消除缩放带来的坐标影响
	float u = x / (float)width;
	float v = y / (float)heigth;


	//如果使用cudaReadModeElementType，则读取uchar4不能转为float
	float4 pixel = tex2D(tex, u, v);		// 这里的坐标时float型，也就说明了为什么可以硬件插值

	char* pix = img + (y*width + x)*channels;
	pix[0] = pixel.x * 255;
	pix[1] = pixel.y * 255;
	pix[2] = pixel.z * 255;
	pix[3] = 0;
}


int main(int argc, char **argv)
{
	Mat img = imread("../demo.jpg", IMREAD_COLOR);
	if (img.empty())
		return 1;
	Mat src = img(cv::Rect(0, 0, 256, 256));
	cvtColor(src, src, CV_BGR2RGBA);

	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();
	int width = cols, height = rows, size = rows * cols*channels;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
	cudaArray *cuArray;
	cudaMallocArray(&cuArray, &channelDesc, width, height);
	cudaMemcpyToArray(cuArray, 0, 0, src.data, size, cudaMemcpyHostToDevice);

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = true;
	cudaBindTextureToArray(tex, cuArray, channelDesc);


#pragma region zoom
	float scale = 4.0;

	Mat zoom_h;
	resize(src, zoom_h, cv::Size(src.rows*scale, src.cols*scale), 0, 0, INTER_LINEAR);

	Mat zoom_d = Mat::zeros(rows*scale, cols*scale, CV_8UC4);
	char *dev_out = nullptr;
	cudaMalloc((void**)&dev_out, zoom_d.rows*zoom_d.cols*zoom_d.channels() * sizeof(uchar));

	dim3 block(32, 8);
	dim3 grid((zoom_d.cols - 1) / block.x + 1, (zoom_d.rows - 1) / block.y + 1);
	zoom_kernel << <grid, block >> > (dev_out, zoom_d.cols, zoom_d.rows, zoom_d.channels());
	cudaError_t err = cudaGetLastError();
	cudaDeviceSynchronize();
	cudaMemcpy(zoom_d.data, dev_out, zoom_d.rows*zoom_d.cols*zoom_d.channels() * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaUnbindTexture(tex);
	cudaFree(dev_out);
	cudaFreeArray(cuArray);

	imshow("orignal", src);
	imshow("zoom_h", zoom_h);
	imshow("zoom_d", zoom_d);
	waitKey(0);
#pragma endregion zoom


	// #pragma region smooth
	// 	Mat out_smooth = Mat::zeros(rows, cols, CV_8UC4);
	// 	char *dev_out = nullptr;
	// 	cudaMalloc((void**)&dev_out, size);
	// 
	// 	dim3 block(16, 16);
	// 	dim3 grid((width - 1) / block.x + 1, (height - 1) / block.y + 1);
	// 	smooth_kernel << <grid, block>> > (dev_out, width, height, channels);
	// 	cudaDeviceSynchronize();
	// 	cudaMemcpy(out_smooth.data, dev_out, size, cudaMemcpyDeviceToHost);
	// 
	// 	cudaUnbindTexture(tex);
	// 	cudaFree(dev_out);
	// 	cudaFreeArray(cuArray);
	// 
	// 	imshow("orignal", src);
	// 	imshow("smooth_image", out_smooth);
	// 	waitKey(0);
	// #pragma endregion smooth

	return 0;
}