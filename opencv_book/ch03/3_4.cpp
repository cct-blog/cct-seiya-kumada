#include <opencv2/opencv.hpp>
#include <filesystem>
#include "3_4.h"

namespace fs = std::filesystem;

void split_plane() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}
	
	auto planes = std::vector<cv::Mat>{};
	cv::split(img, planes);

	cv::imshow("r_plane", planes[2]);
	cv::imshow("g_plane", planes[1]);
	cv::imshow("b_plane", planes[0]);
	cv::imshow("img", img);
	cv::waitKey(0);
}

void merge_plane() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}
	
	auto planes = std::vector<cv::Mat>{};
	cv::split(img, planes);

	auto output = cv::Mat{};
	cv::merge(planes, output);

	cv::imshow("img", img);
	cv::imshow("output", output);
	cv::waitKey(0);
}

void hconcat() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}

	auto hconcat_img = cv::Mat{};
	cv::hconcat(std::vector<cv::Mat>{ img, img }, hconcat_img);
	cv::imshow("hconcat_img", hconcat_img);
	
	cv::waitKey(0);
}

void vconcat() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}

	auto vconcat_img = cv::Mat{};
	cv::vconcat(std::vector<cv::Mat>{ img, img }, vconcat_img);
	cv::imshow("vconcat_img", vconcat_img);
	
	cv::waitKey(0);
}