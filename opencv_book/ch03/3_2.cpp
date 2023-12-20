#include "3_2.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <format>
#include <iostream>
#include <filesystem>
#include <format>
namespace fs = std::filesystem;

void add1() {
	//auto x = std::uint8_t{ 5 };
	//auto y = cv::saturate_cast<std::uint8_t>(5 + 252);
	//std::cout << std::format("y = {}\n", y);

	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto x = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	const auto y = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	const cv::Mat z = x + y;
	std::cout << z.type() << " " << CV_8UC3 << std::endl;
	assert(z.type() == CV_8UC3);
	//cv::imshow("z", z);
	//cv::waitKey(0);
}

void substract1() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto x = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	const auto y = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	const cv::Mat z = 2*x - y;
	assert(z.type() == CV_8UC3);
	cv::imshow("z", z);
	cv::waitKey(0);
}

void bitwise_and() {
	const std::string SRC1_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC1_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	const auto src1 = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);

	const std::string SRC2_PATH = "c:\\projects\\opencv_book\\mask.png"; 
	if (!fs::exists(SRC2_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	const auto src2 = cv::imread(SRC2_PATH, cv::IMREAD_COLOR);

	cv::Mat output;
	cv::bitwise_and(src1, src2, output);

	cv::imshow("output", output);
	cv::waitKey(0);
}

void bitwise_or() {
	const std::string SRC1_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC1_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	const auto src1 = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);

	const std::string SRC2_PATH = "c:\\projects\\opencv_book\\mask.png"; 
	if (!fs::exists(SRC2_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	const auto src2 = cv::imread(SRC2_PATH, cv::IMREAD_COLOR);

	cv::Mat output;
	cv::bitwise_or(src1, src2, output);

	cv::imshow("output", output);
	cv::waitKey(0);
}

void addWeighted() {
	const std::string SRC1_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC1_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	const auto src1 = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);

	auto src2 = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);
	cv::Mat dst{};
	cv::flip(src2, dst, 0);

	auto alpha = float{ 0.5 };
	auto  beta = float{ 0.5 };
	auto gamma = float{ 0.0 };
	auto output = cv::Mat{};

	cv::addWeighted(src1, alpha, dst, beta, gamma, output);

	cv::imshow("output", output);
	cv::waitKey(0);
}

void absdiff() {
	const auto SRC1_PATH = std::string{ "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png" };
	if (!fs::exists(SRC1_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	const auto src1 = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);

	auto src2 = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);
	cv::Mat dst{};
	cv::flip(src2, dst, 0);

	auto output = cv::Mat{};
	cv::absdiff(src1, dst, output);

	cv::imshow("output", output);
	cv::waitKey(0);
}

void copy_image() {
	const auto SRC1_PATH = std::string{ "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png" };
	if (!fs::exists(SRC1_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	auto org_img = cv::imread(SRC1_PATH, cv::IMREAD_COLOR);
	
	auto cv_copy_img = cv::Mat{};
	org_img.copyTo(cv_copy_img);

	//for (auto j = 0; j < org_img.rows; ++j) {
	//	auto ptr = org_img.ptr<cv::Vec3b>(j);
	//	for (auto i = 0; i < org_img.cols; ++i) {
	//		ptr[0] = 0;
	//		ptr[1] = 0;
	//		ptr[2] = 0;
	//	}
	//}


	auto roi = cv::Rect(0, 0, org_img.cols, org_img.rows);
	auto shallow_copy_img = org_img(roi);
	
	cv::rectangle(org_img, cv::Rect(0, 0, 100, 100), cv::Scalar{ 255, 255, 255 }, -1);
	cv::imshow("shallow_copy_img", shallow_copy_img);
	cv::waitKey(0);
}

void minMaxLoc() {
	const auto SRC_PATH = std::string{ "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png" };
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
	auto src = cv::imread(SRC_PATH, cv::IMREAD_GRAYSCALE);

	auto minValue = double{};
	auto maxValue = double{};
	auto minLoc = cv::Point{};
	auto maxLoc = cv::Point{};
	cv::minMaxLoc(src, &minValue, &maxValue, &minLoc, &maxLoc);
	std::cout << std::format("minValue:{}, maxValue:{}, minLoc:{} {}, maxLoc:{} {}\n", minValue, maxValue, minLoc.x, minLoc.y, maxLoc.x, maxLoc.y);
	
	
}
