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

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4\\yolov4.weights" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 設定ファイルを確認する。
	const auto CONFIG_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4\\yolov4.cfg" };
	if (!fs::exists(CONFIG_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 予測モデルを構築する。
	auto model = cv::dnn::DetectionModel(WEIGHTS_PATH, CONFIG_PATH);

	// デバイスとバックエンドを決める。
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// モデルの入力パラメーターを設定する
	auto scale = 1.0 / 255.0;					// スケールファクター
	auto size  = cv::Size{ 416, 416 };			// 入力サイズ
	auto mean  = cv::Scalar{ 0.0, 0.0, 0.0 };	// 差し引かれる平均値
	auto swap  = true;							// チャンネルの順番（True: RGB、False: BGR）
	auto crop  = false;							// クロップ
	model.setInputParams(scale, size, mean, swap, crop);

	// NMSをクラスごとに処理する。
	model.setNmsAcrossClasses(false);

	// COCOファイルパス
	const auto COCO_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\coco.names" };
	if (!fs::exists(COCO_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 全てのクラス名を取得する。
	auto is_good_coco = read_coco_file(COCO_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// クラス名と色を対応させる。
	const auto colors = make_color_table(static_cast<int>(classes.size()));

	// 検出する。
	const auto confidence_threshold = 0.5;
	const auto nms_threshold = 0.4f;
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// 物体を囲む。
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

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\scaled-yolov4\\yolov4x-mish.weights" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 設定ファイルを確認する。
	const auto CONFIG_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\scaled-yolov4\\yolov4x-mish.cfg" };
	if (!fs::exists(CONFIG_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 予測モデルを構築する。
	auto model = cv::dnn::DetectionModel(WEIGHTS_PATH, CONFIG_PATH);

	// デバイスとバックエンドを決める。
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// モデルの入力パラメーターを設定する
	auto scale = 1.0 / 255.0;					// スケールファクター
	auto size  = cv::Size{ 640, 640 };			// 入力サイズ
	auto mean  = cv::Scalar{ 0.0, 0.0, 0.0 };	// 差し引かれる平均値
	auto swap  = true;							// チャンネルの順番（True: RGB、False: BGR）
	auto crop  = false;							// クロップ
	model.setInputParams(scale, size, mean, swap, crop);

	// NMSをクラスごとに処理する。
	model.setNmsAcrossClasses(false);

	// COCOファイルパス
	const auto COCO_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\coco.names" };
	if (!fs::exists(COCO_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 全てのクラス名を取得する。
	auto is_good_coco = read_coco_file(COCO_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// クラス名と色を対応させる。
	const auto colors = make_color_table(static_cast<int>(classes.size()));

	// 検出する。
	const auto confidence_threshold = 0.5;
	const auto nms_threshold = 0.4f;
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// 物体を囲む。
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

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4-tiny\\yolov4-tiny.weights" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 設定ファイルを確認する。
	const auto CONFIG_PATH = std::string{ "C:\\data\\opencv_book\\7.3\\yolov4-tiny\\yolov4-tiny.cfg" };
	if (!fs::exists(CONFIG_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 予測モデルを構築する。
	auto model = cv::dnn::DetectionModel(WEIGHTS_PATH, CONFIG_PATH);

	// デバイスとバックエンドを決める。
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// モデルの入力パラメーターを設定する
	auto scale = 1.0 / 255.0;					// スケールファクター
	auto size  = cv::Size{ 416, 416 };			// 入力サイズ
	auto mean  = cv::Scalar{ 0.0, 0.0, 0.0 };	// 差し引かれる平均値
	auto swap  = true;							// チャンネルの順番（True: RGB、False: BGR）
	auto crop  = false;							// クロップ
	model.setInputParams(scale, size, mean, swap, crop);

	// NMSをクラスごとに処理する。
	model.setNmsAcrossClasses(false);

	// COCOファイルパス
	const auto COCO_PATH = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.3\\yolov4\\coco.names" };
	if (!fs::exists(COCO_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 全てのクラス名を取得する。
	auto is_good_coco = read_coco_file(COCO_PATH);
	if (!is_good_coco) {
		std::cout << "invalid coco file\n";
		return;
	}
	const std::vector<std::string>& classes = is_good_coco.value();

	// クラス名と色を対応させる。
	const auto colors = make_color_table(static_cast<int>(classes.size()));

	// 検出する。
	const auto confidence_threshold = 0.5;
	const auto nms_threshold = 0.4f;
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// 物体を囲む。
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

