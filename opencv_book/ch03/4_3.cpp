#include "4_3.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <format>
namespace fs = std::filesystem;

void thresh() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}
	
	auto out0 = cv::Mat{};
	auto max_value = 255;
	auto thresh = 127;
	cv::threshold(img, out0, thresh, max_value, cv::THRESH_BINARY);
	cv::imshow("THRESH_BINARY", out0);
	
	auto out1 = cv::Mat{};
	cv::threshold(img, out1, thresh, max_value, cv::THRESH_BINARY_INV);
	cv::imshow("THRESH_BINARY_INV", out1);
	
	auto out2 = cv::Mat{};
	cv::threshold(img, out2, thresh, max_value, cv::THRESH_TRUNC);
	cv::imshow("THRESH_TRUNC", out2);

	auto out3 = cv::Mat{};
	cv::threshold(img, out3, thresh, max_value, cv::THRESH_TOZERO);
	cv::imshow("THRESH_TOZERO", out3);
	
	auto out4 = cv::Mat{};
	cv::threshold(img, out4, thresh, max_value, cv::THRESH_TOZERO_INV);
	cv::imshow("THRESH_TOZERO_INV", out4);
	
	cv::waitKey(0);
}

void otsu() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}

	auto out_otsu = cv::Mat{};
	auto otsu_threshold = cv::threshold(img, out_otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	std::cout << std::format("Otsu: {}\n", otsu_threshold);

	auto out_tria = cv::Mat{};
	auto tria_threshold = cv::threshold(img, out_tria, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
	std::cout << std::format("Tria: {}\n", tria_threshold);

	cv::imshow("THRESH_OTSU", out_otsu);
	cv::imshow("THRESH_TRIA", out_tria);
	cv::waitKey(0);
}

void adaptive_thresh_imp() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png";
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		std::cout << "bad file path!\n";
		return;
	}

	auto output = cv::Mat{};
	cv::adaptiveThreshold(img, output, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, 2);

	cv::imshow("Adaptive", output);
	cv::waitKey(0);

	

}
