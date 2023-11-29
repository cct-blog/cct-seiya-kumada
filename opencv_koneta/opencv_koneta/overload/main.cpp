#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
	auto x = cv::Mat{ 3, 3, CV_32FC1, cv::Scalar{1} };
	auto y = cv::Mat{ 3, 3, CV_32FC1, cv::Scalar{2} };

	auto z1 = x + y;
	assert(z1 == 3);

	auto z2 = cv::Mat{};
	cv::add(x, y, z2);
	assert(z2 == 3);

	auto w = x * y; // 行列同士のかけ算
	assert(w == 6);

	auto p = x.dot(y); // 要素同士をかけて和を取る。内積
	assert(p == 18);

	auto q = x / y;
	assert(0.5 == q);

	auto r1 = x - y;
	assert(-1 == r1);

	auto r2 = cv::Mat{};
	cv::subtract(x, y, r2);
	assert(-1 == r2);

	auto s = x.mul(y); //　要素同士のかけ算
	assert(2 == s);

	return 0;
}