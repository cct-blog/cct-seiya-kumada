#include "3_1.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include <format>
namespace fs = std::filesystem;

void blank_gray_image() {
	constexpr int WIDTH { 200 };
	constexpr int HEIGHT{ 100 };
	constexpr int VALUE { 128 };

	// row=HEIGHT、col=WIDTH、画素値ゼロのグレイスケール画像を作成
	auto img1 = cv::Mat{ HEIGHT, WIDTH, CV_8UC1, cv::Scalar{0} };

	// row=HEIGHT、col=WIDTH、画素値128のグレイスケール画像を作成
	auto img2 = cv::Mat{ HEIGHT, WIDTH, CV_8UC1, cv::Scalar{128} };

	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::waitKey(0);
}

void blank_color_image() {
	constexpr int WIDTH	   { 200 };
	constexpr int HEIGHT   { 100 };
	constexpr int CHANNELS { 3 };
	constexpr int RED      { 255 };
	constexpr int GREEN    { 0 };
	constexpr int BLUE     { 0 };

	// row=HEIGHT、col=WIDTH、画素値ゼロのグレイスケール画像を作成
	auto img1 = cv::Mat{ HEIGHT, WIDTH, CV_8UC3, cv::Scalar{0, 0, 0} };

	// row=HEIGHT、col=WIDTH、画素値128のグレイスケール画像を作成
	auto img2 = cv::Mat{ HEIGHT, WIDTH, CV_8UC3, cv::Scalar{BLUE, GREEN, RED} };

	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::waitKey(0);
}

void read_pixel_value() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	const auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	const auto w = img.cols;
	const auto h = img.rows;
	const auto c = img.channels();
	std::cout << std::format("w:{}, h:{}, c:{}\n", w, h, c);
	
	const auto half_w = w / 2;
	const auto half_h = h / 2;
	const auto c0 = 0;
	std::cout << std::format("half_w:{}, half_h:{}, c:{}\n", half_w, half_h, c);

		
	const auto& p = img.at<cv::Vec3b>(half_h, half_w);
	std::cout << std::format("{}, {}, {}\n", p[0], p[1], p[2]);
}

void change_pixel_value() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}

	auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "bad image path!\n";
		return;
	}

	const auto w = img.cols;
	const auto h = img.rows;
	const auto c = img.channels();
	std::cout << std::format("w:{}, h:{}, c:{}\n", w, h, c);
	
	const auto half_w = w / 2;
	const auto half_h = h / 2;
	const auto c0 = 0;
	std::cout << std::format("half_w:{}, half_h:{}, c:{}\n", half_w, half_h, c);

	img.at<cv::Vec3b>(half_h, half_w) = cv::Vec3b{ 0, 0, 0 };
	auto& p = img.at<cv::Vec3b>(half_h, half_w);
	std::cout << std::format("{}, {}, {}\n", p[0], p[1], p[2]);

	p[0] = 100;

	const auto& q = img.at<cv::Vec3b>(half_h, half_w);
	std::cout << std::format("{}, {}, {}\n", q[0], q[1], q[2]);
}

void image_roi() {
	const std::string SRC_PATH = "c:\\Users\\seiya.kumada\\Pictures\\Nv0XgBcV_400x400.png"; 
	if (!fs::exists(SRC_PATH)) {
		std::cout << "bad file path!\n";
		return;
	}
    
	auto img = cv::imread(SRC_PATH, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "bad image path!\n";
		return;
	}

	const int ROI_SIZE = 100;
	const int HALF_WIDTH = ROI_SIZE / 2;
	const int HALF_HEIGHT = ROI_SIZE / 2;

	const int center_x = img.cols / 2;
	const int center_y = img.rows / 2;

	auto roi = cv::Rect(center_x - HALF_WIDTH, center_y - HALF_WIDTH, ROI_SIZE, ROI_SIZE);
	auto center_roi = img(roi);

	//center_roi = cv::Mat{ ROI_SIZE, ROI_SIZE, CV_8UC3, cv::Scalar{0, 0, 0} };
	for (int j = 0; j < center_roi.rows; ++j) {
		auto ptr = center_roi.ptr<cv::Vec3b>(j);
		for (int i = 0; i < center_roi.cols; ++i) {
			ptr[i][0] = 0;
			ptr[i][1] = 0;
			ptr[i][2] = 0;
		}
	}
	cv::imshow("Src Image", img);
	cv::waitKey(0);
}
