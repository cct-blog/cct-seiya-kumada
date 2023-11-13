#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

struct TextPosition {
	std::vector<std::vector<cv::Point>> vertices_;
	std::vector<float>					confidences_;
};

struct RotatedTextPosition {
	std::vector<cv::RotatedRect>	vertices_;
	std::vector<float>				confidences_;
};
class TextDetector {
public:
	TextDetector() = delete;
	TextDetector(const TextDetector&) = delete;
	TextDetector& operator=(const TextDetector&) = delete;

	TextDetector(const std::string& model_path);

	// 画像からテキストを検出する。
	auto detect_vertices(const cv::Mat& image) -> std::optional<TextPosition>;

	// 画像からテキストを検出する。
	auto detect_rotated_rectangles(const cv::Mat& image) -> std::optional<RotatedTextPosition>;

private:
	void init_model(const std::string& model_path);
	std::unique_ptr<cv::dnn::TextDetectionModel_DB> model_;
};


