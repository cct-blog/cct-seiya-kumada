#include "7_5.h"
#include <string>
#include <iostream>
#include <optional>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "util.h"
namespace fs = std::filesystem;



void deeplab_v3() {
	const auto IMAGE_PATH = std::string{"C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.5\\deeplab-v3\\bicycle.jpg"};
	const auto is_good_image = load_image(IMAGE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// �d�݃t�@�C���p�X���m�F����B
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.5\\deeplab-v3\\optimized_graph_voc.pb" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// ���f����ǂݍ��ށB
	auto model = cv::dnn::SegmentationModel(WEIGHTS_PATH);

	// �o�b�N�G���h��ݒ肷��B
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	
	// �^�[�Q�b�g��ݒ肷��B
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// ���f���̓��̓p�����[�^�[��ݒ肷��
	auto scale = 1.0 / 127.5;						// �X�P�[���t�@�N�^�[
	auto size = cv::Size{ 513, 513 };				// ���̓T�C�Y�iVOC�j
	auto mean = cv::Scalar{ 127.5, 127.5, 127.5 };	// ����������镽�ϒl
	auto swap = true;								// �`�����l���̏��ԁiTrue: RGB�AFalse: BGR�j
	auto crop = false;								//�N���b�v
	model.setInputParams(scale, size, mean, swap, crop);

	// VOC�t�@�C���p�X
	const auto VOC_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.5\\deeplab-v3\\voc.names" };
	if (!fs::exists(VOC_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// �S�ẴN���X�����擾����B
	auto is_good_coco = read_coco_file(VOC_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// �N���X���ƐF��Ή�������B
	auto colors = make_color_table(static_cast<int>(classes.size()));
	
	// �w�i�͍��ɂ���B`
	colors[0] = cv::Scalar(0, 0, 0);

	// �Z�O�����e�[�V��������B
	auto mask = cv::Mat{};
	model.segment(image, mask);

	// �}�X�N�ɐF��t����B
	const auto mask_rows = mask.rows;
	const auto mask_cols = mask.cols;
	auto color_mask = cv::Mat(mask_rows, mask_cols, CV_8UC3);
	
	for (auto j = 0; j < mask.rows; ++j) {
		auto mask_p = mask.ptr<cv::uint8_t>(j);
		auto color_mask_p = color_mask.ptr<cv::Vec3b>(j);
		for (auto i = 0; i < mask.cols; ++i) {
			const auto mask_value = static_cast<int>(mask_p[i]);
			const auto& color = colors[mask_value];
			auto& color_mask_value = color_mask_p[i];
			color_mask_value[0] = cv::saturate_cast<cv::uint8_t>(color[0]);
			color_mask_value[1] = cv::saturate_cast<cv::uint8_t>(color[1]);
			color_mask_value[2] = cv::saturate_cast<cv::uint8_t>(color[2]);
		}
	}

	// �}�X�N����͉摜�Ɠ����T�C�Y�ɂ���B
	const auto image_rows = image.rows;
	const auto image_cols = image.cols;
	auto segmented_image = cv::Mat{};
	cv::resize(color_mask, segmented_image, cv::Size(image_cols, image_rows), 0.0, 0.0, cv::INTER_NEAREST);

	// �摜�ƃ}�X�N�����u�����h����B
	//auto alpha = 0.5;
	//auto beta = 1.0 - alpha;
	//auto blended_image = cv::Mat{};
	//cv::addWeighted(image, alpha, segmented_image, beta, 0.0, blended_image);

	cv::imwrite("segmented_image.jpg", segmented_image);
	cv::imshow("segmented_image", segmented_image);
	cv::waitKey(0);
}