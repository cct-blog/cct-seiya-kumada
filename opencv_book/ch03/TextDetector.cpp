#include "TextDetector.h"
#include <opencv2/opencv.hpp>

TextDetector::TextDetector(const std::string& model_path)
	: model_{ nullptr } {
	init_model(model_path);
}

void TextDetector::init_model(const std::string& model_path) {
	// モデルを読み込む。
	model_ = std::make_unique<cv::dnn::TextDetectionModel_DB>(model_path);

	// バックエンドとターゲットを設定する。
	model_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// モデルの入力パラメーターを設定する
	const auto scale = 1.0 / 255.0;												// スケールファクター
	//const auto size = cv::Size{ 736, 736 };									// 入力サイズ（MSRA - TD500）
	const auto size = cv::Size{ 736, 1280 };									// 入力サイズ（ICDAR2015）
	const auto mean = cv::Scalar{ 122.67891434, 116.66876762, 104.00698793 };	// 差し引かれる平均値
	const auto swap = false;													// チャンネルの順番（True: RGB、False: BGR）
	const auto crop = false;													// クロップ
	model_->setInputParams(scale, size, mean, swap, crop);

	// テキスト検出のパラメーターを設定する
	const auto binary_threshold = 0.3;	// 二値化の閾値
	const auto polygon_threshold = 0.5;	// テキスト輪郭スコアの閾値
	const auto max_candidates = 200;	//テキスト候補領域の上限値
	const auto unclip_ratio = 2.0;		// アンクリップ率
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
