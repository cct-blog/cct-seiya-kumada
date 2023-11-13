#include "TextRecognizer.h"
#include "util.h"

TextRecognizer::TextRecognizer(const std::string& model_path, const std::string& vocabulary_path)
	: model_{ nullptr } {
	init_model(model_path, vocabulary_path);
}


auto TextRecognizer::read_vocabularies(const std::string& path) -> std::vector<std::string> {
	return std::vector<std::string>{ };
}

void TextRecognizer::init_model(const std::string& model_path, const std::string& vocabulary_path) {
	auto pos = model_path.find("DenseNet_CTC.onnx");
	if (pos == std::string::npos) {
		requires_gray_ = false;
	}
	else {
		requires_gray_ = true;
	}

	// モデルを読み込む。
	model_ = std::make_unique<cv::dnn::TextRecognitionModel>(model_path);

	// バックエンドとターゲットを設定する。
	model_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// モデルの入力パラメーターを設定する
	const auto scale = 1.0 / 127.5;											// スケールファクター
	const auto size = cv::Size{ 100, 32 };									// 入力サイズ（ICDAR2015）
	const auto mean = cv::Scalar{ 127.5, 127.5, 127.5 }; // 差し引かれる平均値
	const auto swap = true;													// チャンネルの順番（True: RGB、False: BGR）
	const auto crop = false;													// クロップ
	model_->setInputParams(scale, size, mean, swap, crop);

	// デコードタイプを設定する
	auto type = std::string{ "CTC-greedy" };            // 貪欲法
	model_->setDecodeType(type);

	// 語彙リストを設定する
	auto vocabularies = read_coco_file(vocabulary_path);
	if (vocabularies) {
		model_->setVocabulary(vocabularies.value());
	}
	else {
		throw std::runtime_error("invalid vocabulary file");
	}
}


auto TextRecognizer::recognize(const cv::Mat& image) -> std::optional<std::string> {
	if (model_ == nullptr) {
		return std::nullopt;
	}

	if (image.empty()) {
		return std::nullopt;
	}

	auto type = image.type();
	cv::Mat updated_image = image.clone();
	if (type != CV_8UC1 && requires_gray_) {
		cv::cvtColor(image, updated_image, cv::COLOR_BGR2GRAY);
	}

	return model_->recognize(updated_image);
}
