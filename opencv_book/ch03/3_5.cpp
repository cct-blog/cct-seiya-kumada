#include "3_5.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <format>
#include <ranges>

namespace fs = std::filesystem;
namespace vs = std::views;

void getBuildInformation() {
	std::cout << cv::getBuildInformation() << std::endl;
}

void measurement_time_once() {
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

	auto timer = cv::TickMeter();
	timer.start();
	for (auto i = 0; i < 1000; ++i) {
		auto gray = cv::Mat{};
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	}

	timer.stop();
	auto measurement_time = timer.getTimeMilli();
	std::cout << measurement_time << std::endl;
	std::cout << std::format("measurement_time:{:.3f}ms\n", measurement_time);
}

void measurement_time() {
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

	auto timer = cv::TickMeter();
	for (auto i : vs::iota(0, 5)) {
		timer.reset();
		timer.start();
		auto gray = cv::Mat{};
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		timer.stop();
		auto measurement_time = timer.getTimeMilli();
		std::cout << std::format("{}: measurement_time:{:.3f}ms\n", i, measurement_time);
	}
}

void write_xml() {
	auto filename = std::string{"output.xml"};
	auto fs = cv::FileStorage{ filename, cv::FileStorage::WRITE };
	if (!fs.isOpened()) {
		std::cout << "Failed to load XML file.\n";
		return;
	}

	auto R = cv::Mat::eye(cv::Size(3, 3), CV_8UC1);
	auto T = cv::Mat::zeros(cv::Size(3, 1), CV_8UC1);

	fs.write("R", R);
	fs.write("T", T);

	fs.writeComment("This is comment");
}

void read_xml() {
	auto filename = std::string{"c:/projects/opencv_book/opencv_book/ch03/output.xml"};
	if (!fs::exists(filename)) {
		std::cout << "bad file\n";
		return;
	}
	auto fs = cv::FileStorage(filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << "Failed to load XML file.\n";
		return;
	}
	auto R = cv::Mat{};
	fs["R"] >> R;
	
	auto T = cv::Mat{};
	fs["T"] >> T;

	std::cout << R << std::endl;
	std::cout << T << std::endl;
}

void useOptimized() {
	std::cout << std::ios::boolalpha;
	//std::cout << cv::useOptimized() << std::endl; リンクできない。

	//cv::setUseOptimized(true); これもリンクできない。
}