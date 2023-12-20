#include "3_3.h"
#include <vector>
#include <opencv2/opencv.hpp>

void mean_value() {
	auto img = std::vector<int>{ 1, 2, 3, 4, 5 };
	auto mean = cv::mean(img);
	std::cout << std::format("{}\n", mean[0]);
	std::cout << mean.channels << std::endl;
}

void sum_pixel_value() {
	auto img = std::vector<int>{ 1, 2, 3, 4, 5 };
	auto mean = cv::sum(img);
	std::cout << std::format("{}\n", mean[0]);
	std::cout << mean.channels << std::endl;
}

void countNonZero() {
	auto img = std::vector<int>{ 0, 1, 0, 2, 3, 4, 5 };
	auto c = cv::countNonZero(img);
	assert(c == 2);
}
