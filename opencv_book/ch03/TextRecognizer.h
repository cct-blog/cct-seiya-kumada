#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>

class TextRecognizer {
public:
	TextRecognizer() = delete;
	TextRecognizer(const TextRecognizer&) = delete;
	TextRecognizer& operator=(const TextRecognizer&) = delete;
	
	TextRecognizer(const std::string& model_path, const std::string& vocabulary_path);
	auto recognize(const cv::Mat& image) -> std::optional<std::string>;

private:
	void init_model(const std::string& model_path, const std::string& vocabulary_path);
	std::unique_ptr<cv::dnn::TextRecognitionModel> model_;
	bool requires_gray_;

	auto read_vocabularies(const std::string& path) -> std::vector<std::string>;
};
