#include <string>
#include <optional>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <random>
#include <ranges>
namespace fs = std::filesystem;
namespace vs = std::views;


auto load_image(const std::string& path) -> std::optional<cv::Mat> {
	// 画像のパスを確認する。
	if (!fs::exists(path)) {
		std::cout << "invalid file path\n";
		return std::nullopt;
	}

	// 画像を読み込む。
	const auto image = cv::imread(path, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "invalid file\n";
		return std::nullopt;
	}
	return image;
}

auto read_coco_file(const std::string& path) -> std::optional<std::vector<std::string>> {
	auto ifs = std::ifstream{ path };
	if (!ifs.is_open()) {
		return std::nullopt;
	}

	auto line = std::string{};
	auto lines = std::vector<std::string>{};
	while (std::getline(ifs, line)) {
		boost::algorithm::trim(line);
		lines.emplace_back(line);
	}
	return lines;
}

auto make_color_table(int num) -> std::vector<cv::Scalar> {
	// メルセンヌ・ツイスター法による擬似乱数生成器を、
	// ハードウェア乱数をシードにして初期化
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());

	// 一様分布
	// [-1.0f, 1.0f)の値の範囲で、等確率に実数を生成する
	std::uniform_int_distribution<int> dist(0, 256);

	auto table = std::vector<cv::Scalar> {};
	for (auto _ : vs::iota(0, num)) {
		auto r = dist(engine);
		auto g = dist(engine);
		auto b = dist(engine);
		table.emplace_back(b, g, r);
	}

	return table;
}