#include <opencv2/opencv.hpp>
// https://qiita.com/yoya/items/bfef7404ded22649a2af

int main(int argc, char* argv[]) {
	const auto IMAGE_PATH = std::string{"C:/data/cct_blog/opencv_koneta/icelandlake2540.jpg"};
	const auto image = cv::imread(IMAGE_PATH);
	
	cv::Vec3b pixel = image.at<cv::Vec3b>(0, 0);
	
	// B,G,R�̏��ɓ����Ă���B 
	uchar blue = pixel[0];
	uchar green = pixel[1];
	uchar red = pixel[2];

	// OpenCV�̑O�g�ł���IPL(Intel Image Processing Library)�ł̕��т�BGR�ł���������B
	// IPL�͌��XWindows�p�ł���AWindows�ł͓����摜�`���Ƃ���DIB(Device ndows Independent Bitmap)���g���Ă����B
	// DIB�ł̉�f�̕��т�BGR�ł���B

	return 0;
}