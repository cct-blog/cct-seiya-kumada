#include "TextDetector.h"
#include <opencv2/opencv.hpp>

TextDetector::TextDetector(const std::string& model_path)
	: model_{ nullptr } {
	init_model(model_path);
}

void TextDetector::init_model(const std::string& model_path) {
	// ���f����ǂݍ��ށB
	model_ = std::make_unique<cv::dnn::TextDetectionModel_DB>(model_path);

	// �o�b�N�G���h�ƃ^�[�Q�b�g��ݒ肷��B
	model_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// ���f���̓��̓p�����[�^�[��ݒ肷��
	const auto scale = 1.0 / 255.0;												// �X�P�[���t�@�N�^�[
	//const auto size = cv::Size{ 736, 736 };									// ���̓T�C�Y�iMSRA - TD500�j
	const auto size = cv::Size{ 736, 1280 };									// ���̓T�C�Y�iICDAR2015�j
	const auto mean = cv::Scalar{ 122.67891434, 116.66876762, 104.00698793 };	// ����������镽�ϒl
	const auto swap = false;													// �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
	const auto crop = false;													// �N���b�v
	model_->setInputParams(scale, size, mean, swap, crop);

	// �e�L�X�g���o�̃p�����[�^�[��ݒ肷��
	const auto binary_threshold = 0.3;	// ��l����臒l
	const auto polygon_threshold = 0.5;	// �e�L�X�g�֊s�X�R�A��臒l
	const auto max_candidates = 200;	//�e�L�X�g���̈�̏���l
	const auto unclip_ratio = 2.0;		// �A���N���b�v��
	model_->setBinaryThreshold(binary_threshold);
	model_->setPolygonThreshold(polygon_threshold);
	model_->setMaxCandidates(max_candidates);
	model_->setUnclipRatio(unclip_ratio);
}


auto TextDetector::detect_vertices(const cv::Mat& image) -> std::optional<TextPosition> {
	if (model_ == nullptr) {
		return std::nullopt;
	}

	if (image.empty()) {
		return std::nullopt;
	}

	auto text_position = TextPosition{};
	model_->detect(image, text_position.vertices_, text_position.confidences_);
	return text_position;
}

auto TextDetector::detect_rotated_rectangles(const cv::Mat& image) -> std::optional<RotatedTextPosition> {
	if (model_ == nullptr) {
		return std::nullopt;
	}

	if (image.empty()) {
		return std::nullopt;
	}

	auto text_position = RotatedTextPosition{};
	model_->detectTextRectangles(image, text_position.vertices_, text_position.confidences_);
	return text_position;
}
