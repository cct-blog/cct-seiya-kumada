#include <opencv2/opencv.hpp>
// https://qiita.com/yoya/items/bfef7404ded22649a2af

int main(int argc, char* argv[]) {
	const auto IMAGE_PATH = std::string{"C:/data/cct_blog/opencv_koneta/icelandlake2540.jpg"};
	const auto image = cv::imread(IMAGE_PATH);
	
	cv::Vec3b pixel = image.at<cv::Vec3b>(0, 0);
	
	// B,G,Rの順に入っている。 
	uchar blue = pixel[0];
	uchar green = pixel[1];
	uchar red = pixel[2];

	// OpenCVの前身であるIPL(Intel Image Processing Library)での並びがBGRであったから。
	// IPLは元々Windows用であり、Windowsでは内部画像形式としてDIB(Device ndows Independent Bitmap)が使われていた。
	// DIBでの画素の並びはBGRである。

	return 0;
}