#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <valarray>
#include <algorithm>
#include <stack>
#include <ctime>
#include <cmath>

#define original 0
#define translation 1
#define zoom 2
#define rotate 3

void averageAtMyMat(cv::Mat &mat)
{
	std::valarray<float> va1(9);
	std::valarray<float> va2[3];
	va2[0].resize(9);	va2[1].resize(9);	va2[2].resize(9);
	if (mat.channels() == 1)
	{
		for (int i = 1; i < mat.rows-1; i++)
		{
			int v = 0;
			for (int j = 1; j < mat.cols - 1; j++)
			{
				for (int t = i - 1; t < i + 2; t++)
				{
					for (int u = j - 1; u < j + 2; u++)
					{
						va1[v] = mat.at<uchar>(t, u);
						if (v == 4)
						{
							va1[v] = mat.at<uchar>(t, u) * 2;
						}
						v++;
					}
				}
				int result = va1.sum() / 10;
				result = (result > 255 ? 255 : result);
				v = 0;
				mat.at<uchar>(i, j) = (uchar)result;
			}
		}
	}
	else if (mat.channels() == 3)
	{
		for (int i = 1; i < mat.rows - 1; i++)
		{
			int v = 0;
			for (int j = 1; j < mat.cols - 1; j++)
			{
				for (int t = i - 1; t < i + 2; t++)
				{
					for (int u = j - 1; u < j + 2; u++,v++)
					{
						va2[0][v] = mat.at<cv::Vec3b>(t, u)[0];
						va2[1][v] = mat.at<cv::Vec3b>(t, u)[1];
						va2[2][v] = mat.at<cv::Vec3b>(t, u)[2];
						if (v == 4)
						{
							va2[0][v] = mat.at<cv::Vec3b>(t, u)[0] * 2;
							va2[1][v] = mat.at<cv::Vec3b>(t, u)[1] * 2;
							va2[2][v] = mat.at<cv::Vec3b>(t, u)[2] * 2;
						}
					}
				}
				int f1 = va2[0].sum() / 10;
				f1 = (f1 > 255 ? 255 : f1);
				int f2 = va2[1].sum() / 10;
				f2 = (f2 > 255 ? 255 : f2);
				int f3 = va2[2].sum() / 10;
				f3 = (f3 > 255 ? 255 : f3);
				v = 0;
				mat.at<cv::Vec3b>(i, j)[0] = uchar(f1);
				mat.at<cv::Vec3b>(i, j)[1] = uchar(f2);
				mat.at<cv::Vec3b>(i, j)[2] = uchar(f3);
			}
		}
	}
	else
	{
		std::cerr << "Bad Image!" << std::endl;
		return ;
	}
}

cv::Mat ImageRotate(cv::Mat &src, double degree)
{
	cv::Mat dest;
	double degree_rad = degree / 180.0;
	int new_width = round(fabs(src.rows*cos(degree_rad)) + fabs(src.cols*sin(degree_rad)));
	int new_height = round(fabs(src.cols*cos(degree_rad)) + fabs(src.rows*cos(degree_rad)));
	if (src.channels() == 1)
		dest = cv::Mat::zeros(new_width, new_height, CV_8UC1);
	else
		dest = cv::Mat::zeros(new_width, new_height, CV_8UC3);

	return dest;
}

/*
*	AX= B
*	X = B*A^-1
*/

cv::Mat ImageTranslation(cv::Mat &src, double val_x, double val_y)
{
	cv::Mat dest;
	if (src.channels() == 1)
		dest = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	else
		dest = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
	cv::Mat mat = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, val_x, val_y,1);
	cv::Mat reverse_mat = mat.inv();	//逆矩阵
	for (int i = 0; i < dest.rows; i++)
	{
		for (int j = 0; j < dest.cols; j++)
		{
			cv::Mat dest_coordinates = (cv::Mat_<double>(1, 3) << j,i,1);
			cv::Mat src_coordinates = dest_coordinates * reverse_mat;
			double v = src_coordinates.at<double>(0,0);
			double w = src_coordinates.at<double>(0,1);
			if (v >= 0 && w >= 0 && v <= src.cols - 1 && w <= src.rows - 1)
			{
				int top = floor(w), bottom = ceil(w), left = floor(v), right = ceil(v); //与映射到原图坐标相邻的四个像素点的坐标
				double pw = w - top;
				double pv = v - left; 
				if (src.channels() == 1) {
					//灰度图像
					dest.at<uchar>(i, j) = (1 - pw)*(1 - pv)*src.at<uchar>(top, left) + (1 - pw)*pv*src.at<uchar>(top, right) + pw * (1 - pv)*src.at<uchar>(bottom, left) + pw * pv*src.at<uchar>(bottom, right);
				}
				else 
				{
					//彩色图像
					dest.at<cv::Vec3b>(i, j)[0] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[0] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[0] + pw * (1 - pv)*src.at<cv::Vec3b>(bottom, left)[0] + pw * pv*src.at<cv::Vec3b>(bottom, right)[0];
					dest.at<cv::Vec3b>(i, j)[1] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[1] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[1] + pw * (1 - pv)*src.at<cv::Vec3b>(bottom, left)[1] + pw * pv*src.at<cv::Vec3b>(bottom, right)[1];
					dest.at<cv::Vec3b>(i, j)[2] = (1 - pw)*(1 - pv)*src.at<cv::Vec3b>(top, left)[2] + (1 - pw)*pv*src.at<cv::Vec3b>(top, right)[2] + pw * (1 - pv)*src.at<cv::Vec3b>(bottom, left)[2] + pw * pv*src.at<cv::Vec3b>(bottom, right)[2];
				}
			}
		}
	}
	return dest;
}

cv::Mat ImageZoom(cv::Mat &src, double zoomrate)
{
	return src;
}

void MakeNoisy2(cv::Mat &mat, int n)
{
	srand(uint32_t(time(0)));
	for (int i = 0; i < n; i++)
	{
		int px = rand() % mat.rows;
		int py = rand() % mat.cols;
		uchar flag = 0;
		if ((px + py) % 2 != 0)
			flag = 255;
		else
			flag = 0;
		if (mat.channels() == 1)
		{
			uchar *pt = mat.ptr<uchar>(px);
			pt[py] = 0;
		}
		else if (mat.channels() == 3)
		{
			uchar *pt = mat.ptr<uchar>(px, py);
			pt[0] = 0;
			pt[1] = 0;
			pt[2] = 0;
		}
	}
}

//mat 目标图像	n 循环次数
void MakeNoisy(cv::Mat &mat, int n)
{
	srand(uint32_t(time(nullptr)));
	for (int i = 0; i < n; i++)
	{
		int pos1 = rand() % mat.rows;
		int pos2 = rand() % mat.cols;
		int flag = 0;
		if ((pos1 + pos2) % 2 != 0)
			flag = 255;
		else
			flag = 0;
		if (mat.channels() == 1)
		{
			//处理灰度
			mat.at<uchar>(pos1, pos2) = flag;
		}
		else
		{
			//处理彩色
			mat.at<cv::Vec3b>(pos1, pos2)[0] = flag;
			mat.at<cv::Vec3b>(pos1, pos2)[1] = flag;
			mat.at<cv::Vec3b>(pos1, pos2)[2] = flag;
		}
	}
}
