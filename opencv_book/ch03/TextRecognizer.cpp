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

	// ���f����ǂݍ��ށB
	model_ = std::make_unique<cv::dnn::TextRecognitionModel>(model_path);

	// �o�b�N�G���h�ƃ^�[�Q�b�g��ݒ肷��B
	model_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// ���f���̓��̓p�����[�^�[��ݒ肷��
	const auto scale = 1.0 / 127.5;											// �X�P�[���t�@�N�^�[
	const auto size = cv::Size{ 100, 32 };									// ���̓T�C�Y�iICDAR2015�j
	const auto mean = cv::Scalar{ 127.5, 127.5, 127.5 }; // ����������镽�ϒl
	const auto swap = true;													// �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
	const auto crop = false;													// �N���b�v
	model_->setInputParams(scale, size, mean, swap, crop);

	// �f�R�[�h�^�C�v��ݒ肷��
	auto type = std::string{ "CTC-greedy" };            // �×~�@
	model_->setDecodeType(type);

	// ��b���X�g��ݒ肷��
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
