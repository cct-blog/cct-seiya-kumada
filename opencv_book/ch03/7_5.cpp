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

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.5\\deeplab-v3\\optimized_graph_voc.pb" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// モデルを読み込む。
	auto model = cv::dnn::SegmentationModel(WEIGHTS_PATH);

	// バックエンドを設定する。
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	
	// ターゲットを設定する。
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// モデルの入力パラメーターを設定する
	auto scale = 1.0 / 127.5;						// スケールファクター
	auto size = cv::Size{ 513, 513 };				// 入力サイズ（VOC）
	auto mean = cv::Scalar{ 127.5, 127.5, 127.5 };	// 差し引かれる平均値
	auto swap = true;								// チャンネルの順番（True: RGB、False: BGR）
	auto crop = false;								//クロップ
	model.setInputParams(scale, size, mean, swap, crop);

	// VOCファイルパス
	const auto VOC_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.5\\deeplab-v3\\voc.names" };
	if (!fs::exists(VOC_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 全てのクラス名を取得する。
	auto is_good_coco = read_coco_file(VOC_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// クラス名と色を対応させる。
	auto colors = make_color_table(static_cast<int>(classes.size()));
	
	// 背景は黒にする。`
	colors[0] = cv::Scalar(0, 0, 0);

	// セグメンテーションする。
	auto mask = cv::Mat{};
	model.segment(image, mask);

	// マスクに色を付ける。
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

	// マスクを入力画像と同じサイズにする。
	const auto image_rows = image.rows;
	const auto image_cols = image.cols;
	auto segmented_image = cv::Mat{};
	cv::resize(color_mask, segmented_image, cv::Size(image_cols, image_rows), 0.0, 0.0, cv::INTER_NEAREST);

	// 画像とマスクをαブレンドする。
	//auto alpha = 0.5;
	//auto beta = 1.0 - alpha;
	//auto blended_image = cv::Mat{};
	//cv::addWeighted(image, alpha, segmented_image, beta, 0.0, blended_image);

	cv::imwrite("segmented_image.jpg", segmented_image);
	cv::imshow("segmented_image", segmented_image);
	cv::waitKey(0);
}