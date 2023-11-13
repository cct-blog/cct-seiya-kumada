#include "7_3.h"
#include "7_3.h"
#include <string>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <fstream>
#include <algorithm>
#include <boost/algorithm/string/trim.hpp>
#include <random>
#include <ranges>
#include <boost/range/combine.hpp>
#include "util.h"

namespace vs = std::views;
namespace fs = std::filesystem;





void yolov4() {
	const auto FILE_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\dog.jpg" };
	auto is_good_image = load_image(FILE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// �d�݃t�@�C���p�X���m�F����B
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4\\yolov4.weights" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �ݒ�t�@�C�����m�F����B
	const auto CONFIG_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4\\yolov4.cfg" };
	if (!fs::exists(CONFIG_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �\�����f�����\�z����B
	auto model = cv::dnn::DetectionModel(WEIGHTS_PATH, CONFIG_PATH);

	// �f�o�C�X�ƃo�b�N�G���h�����߂�B
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// ���f���̓��̓p�����[�^�[��ݒ肷��
	auto scale = 1.0 / 255.0;					// �X�P�[���t�@�N�^�[
	auto size  = cv::Size{ 416, 416 };			// ���̓T�C�Y
	auto mean  = cv::Scalar{ 0.0, 0.0, 0.0 };	// ����������镽�ϒl
	auto swap  = true;							// �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
	auto crop  = false;							// �N���b�v
	model.setInputParams(scale, size, mean, swap, crop);

	// NMS���N���X���Ƃɏ�������B
	model.setNmsAcrossClasses(false);

	// COCO�t�@�C���p�X
	const auto COCO_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\coco.names" };
	if (!fs::exists(COCO_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �S�ẴN���X�����擾����B
	auto is_good_coco = read_coco_file(COCO_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// �N���X���ƐF��Ή�������B
	const auto colors = make_color_table(static_cast<int>(classes.size()));

	// ���o����B
	const auto confidence_threshold = 0.5;
	const auto nms_threshold = 0.4f;
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// ���̂��͂ށB
	for (const auto& p : boost::combine(classIds, boxes)) {
		const auto& cls = boost::get<0>(p);
		const auto& box = boost::get<1>(p);
		const auto& color = colors[cls];
		cv::rectangle(image, box, color, 1, cv::LINE_AA);
		const auto& name = classes[cls];
		cv::putText(image, name, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
	}

	cv::imshow("object detection", image);
	cv::waitKey(0);
}

void scaled_yolov4() {
	const auto FILE_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\dog.jpg" };
	auto is_good_image = load_image(FILE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// �d�݃t�@�C���p�X���m�F����B
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\scaled-yolov4\\yolov4x-mish.weights" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �ݒ�t�@�C�����m�F����B
	const auto CONFIG_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\scaled-yolov4\\yolov4x-mish.cfg" };
	if (!fs::exists(CONFIG_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �\�����f�����\�z����B
	auto model = cv::dnn::DetectionModel(WEIGHTS_PATH, CONFIG_PATH);

	// �f�o�C�X�ƃo�b�N�G���h�����߂�B
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// ���f���̓��̓p�����[�^�[��ݒ肷��
	auto scale = 1.0 / 255.0;					// �X�P�[���t�@�N�^�[
	auto size  = cv::Size{ 640, 640 };			// ���̓T�C�Y
	auto mean  = cv::Scalar{ 0.0, 0.0, 0.0 };	// ����������镽�ϒl
	auto swap  = true;							// �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
	auto crop  = false;							// �N���b�v
	model.setInputParams(scale, size, mean, swap, crop);

	// NMS���N���X���Ƃɏ�������B
	model.setNmsAcrossClasses(false);

	// COCO�t�@�C���p�X
	const auto COCO_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\coco.names" };
	if (!fs::exists(COCO_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �S�ẴN���X�����擾����B
	auto is_good_coco = read_coco_file(COCO_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// �N���X���ƐF��Ή�������B
	const auto colors = make_color_table(static_cast<int>(classes.size()));

	// ���o����B
	const auto confidence_threshold = 0.5;
	const auto nms_threshold = 0.4f;
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// ���̂��͂ށB
	for (const auto& p : boost::combine(classIds, boxes)) {
		const auto& cls = boost::get<0>(p);
		const auto& box = boost::get<1>(p);
		const auto& color = colors[cls];
		cv::rectangle(image, box, color, 1, cv::LINE_AA);
		const auto& name = classes[cls];
		cv::putText(image, name, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
	}

	cv::imshow("object detection", image);
	cv::waitKey(0);
}

void tiny_yolov4() {
	const auto FILE_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\dog.jpg" };
	auto is_good_image = load_image(FILE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// �d�݃t�@�C���p�X���m�F����B
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4-tiny\\yolov4-tiny.weights" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �ݒ�t�@�C�����m�F����B
	const auto CONFIG_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4-tiny\\yolov4-tiny.cfg" };
	if (!fs::exists(CONFIG_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �\�����f�����\�z����B
	auto model = cv::dnn::DetectionModel(WEIGHTS_PATH, CONFIG_PATH);

	// �f�o�C�X�ƃo�b�N�G���h�����߂�B
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// ���f���̓��̓p�����[�^�[��ݒ肷��
	auto scale = 1.0 / 255.0;					// �X�P�[���t�@�N�^�[
	auto size  = cv::Size{ 416, 416 };			// ���̓T�C�Y
	auto mean  = cv::Scalar{ 0.0, 0.0, 0.0 };	// ����������镽�ϒl
	auto swap  = true;							// �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
	auto crop  = false;							// �N���b�v
	model.setInputParams(scale, size, mean, swap, crop);

	// NMS���N���X���Ƃɏ�������B
	model.setNmsAcrossClasses(false);

	// COCO�t�@�C���p�X
	const auto COCO_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\coco.names" };
	if (!fs::exists(COCO_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �S�ẴN���X�����擾����B
	auto is_good_coco = read_coco_file(COCO_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// �N���X���ƐF��Ή�������B
	const auto colors = make_color_table(static_cast<int>(classes.size()));

	// ���o����B
	const auto confidence_threshold = 0.5;
	const auto nms_threshold = 0.4f;
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// ���̂��͂ށB
	for (const auto& p : boost::combine(classIds, boxes)) {
		const auto& cls = boost::get<0>(p);
		const auto& box = boost::get<1>(p);
		const auto& color = colors[cls];
		cv::rectangle(image, box, color, 1, cv::LINE_AA);
		const auto& name = classes[cls];
		cv::putText(image, name, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
	}

	cv::imshow("object detection", image);
	cv::waitKey(0);
}

