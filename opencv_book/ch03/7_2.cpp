#include "7_2.h"
#include <string>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/range/combine.hpp>
namespace fs = std::filesystem;

void haarcascade() {
	const auto FILE_PATH = std::string{"c:/data/opencv_book/sample_01.jpg"};
	if (!fs::exists(FILE_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	auto image = cv::imread(FILE_PATH, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "invalid file\n";
		return;
	}

	const auto MODEL_PATH = std::string{"c:/data/opencv_book/7.2/7.2/haarcascade/haarcascade_frontalface_default.xml"};
	auto cascade = cv::CascadeClassifier(MODEL_PATH);
	if (cascade.empty()) {
		std::cout << "invalid model\n";
		return;
	}

	auto gray = cv::Mat{};
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	
	const auto h = gray.rows;
	const auto w = gray.cols;

	const auto min_h = h / 10;
	const auto min_w = w / 10;

	std::vector<cv::Rect> boxes{};
	cascade.detectMultiScale(gray, boxes, 1.1 /*scale factor*/, 1 /*min neighbors*/, 0, cv::Size(min_w, min_h));

	for (const auto& box : boxes) {
		cv::rectangle(image, box, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
	}

	cv::imshow("face detection", image);
	cv::waitKey(0);
}


void opencv_face_detector() {
	// パスの有無を確認する。
	const auto FILE_PATH = std::string{"c:/data/opencv_book/physics_color.jpg"};
	if (!fs::exists(FILE_PATH)) {
		std::cout << "invalid file path\n";
		return;
	}

	// 画像を読み込む。
	auto image = cv::imread(FILE_PATH, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "invalid file\n";
		return;
	}

	// 予測モデルを読み込む。
	const auto weights_path = std::string{"c:/data/opencv_book/7.2/7.2/opencv_face_detector/opencv_face_detector_fp16.caffemodel"};
	const auto config_path = std::string{ "c:/data/opencv_book/7.2/7.2/opencv_face_detector/opencv_face_detector_fp16.prototxt" };
	auto model = cv::dnn::DetectionModel{ weights_path, config_path };
	//auto net = cv::dnn::readNet(weights_path, config_path);
	//auto model = cv::dnn::DetectionModel(net);
	
	// バックエンドとデバイスを設定する。
	model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	const auto scale = 1.0; // スケールファクター
	const auto size = cv::Size(1600, 806); // 入力サイズ
	const auto mean = cv::Scalar(104.0, 177.0, 123.0); // 差し引かれる平均値
	const auto swap = false; // BGR, true -> RGB
	const auto crop = false;

	// モデルのパラメータを決める。
	model.setInputParams(scale, size, mean, swap, crop);
	//model.setInputScale(scale)
	//	 .setInputSize(size)
	//	 .setInputMean(mean)
	//	 .setInputSwapRB(swap)
	//	 .setInputCrop(crop);

	// 顔を検出する。
	auto classIds = std::vector<int>{};
	auto confidences = std::vector<float>{};
	auto boxes = std::vector<cv::Rect>{};
	auto confidence_threshold = 0.6f;
	auto nms_threshold = 0.4f;
	model.detect(image, classIds, confidences, boxes, confidence_threshold, nms_threshold);

	// 顔を囲む。
	for (const auto& box : boxes) {
		cv::rectangle(image, box, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
	}

	std::cout << std::endl;
	for (const auto& p : boost::combine(classIds, confidences)) {
		auto c = boost::get<0>(p);
		auto f = boost::get<1>(p);
		std::cout << std::format("class:{}, confidence:{}", c, f) << " ";
	}
	std::cout << std::endl;
	cv::imshow("face detection", image);
	auto output_path = "C:\\data\\opencv_book\\face_detetion_result.jpg";
	cv::imwrite(output_path, image);
	cv::waitKey(0);
	
}
