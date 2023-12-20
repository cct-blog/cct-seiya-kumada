#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>


int main(int argc, char* argv[]) {
	auto x = cv::Mat{ 3, 3, CV_32FC1, cv::Scalar{1} };
	auto y = cv::Mat{ 3, 3, CV_32FC1, cv::Scalar{2} };

	// ==
	{
		assert(x == 1 && y == 2);
		assert(1 == x && 2 == yy);
	}

	// + 
	{
		auto z = x + y;
		auto w = 1 + y;
		assert(z == 3);
		assert(w == 3);
	}

	// -
	{
		auto z = x - y;
		auto w = 1 - y;
		assert(z == -1);
		assert(w == -1);
	}

	return 0;
}