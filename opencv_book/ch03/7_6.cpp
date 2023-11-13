#include "7_6.h"
#include <string>
#include "util.h"
#include <iostream>
#include "TextDetector.h"
#include "TextRecognizer.h"
#include <filesystem>
#include <boost/range/combine.hpp>
namespace fs = std::filesystem;

void db() {
	// ファイルを読み込む。
	const auto IMAGE_PATH = std::string{"C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.6\\db\\text.jpg"};
	const auto is_good_image = load_image(IMAGE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.6\\db\\DB_IC15_resnet50.onnx" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid model path\n";
		return;
	}

	// 検出器を作る。
	auto detector = TextDetector{ WEIGHTS_PATH };

	// テキストを検出する。
	auto text_info = detector.detect_vertices(image);
	//auto rotated_text_info = detector.detect_rotated_rectangles(image);
	if (!text_info) {
		std::cout << "invalid detection\n";
		return;
	}
	const auto& vertices = text_info.value().vertices_;
	const auto& confidences = text_info.value().confidences_;

	// 検出したテキスト領域を描画する。
	for (const auto& vertex : vertices) {
		auto is_closed = true;
		auto color = cv::Scalar{ 0, 255, 0 };
		auto thickness = 2;
		cv::polylines(image, vertex, is_closed, color, thickness, cv::LINE_AA);
	}
	cv::imshow("image", image);
	cv::waitKey(0);
	cv::imwrite("detected_image.jpg", image);
}

auto get_text_images(
	const cv::Mat& image, 
	const std::vector<std::vector<cv::Point>>& vertices) 
-> std::vector<cv::Mat> {
	auto text_images = std::vector<cv::Mat>{};
	auto size = cv::Size{ 100, 32 }; // w, h
	
	text_images.reserve(vertices.size());
	for (const auto& vertex : vertices) {
		auto target_points = std::vector<cv::Point>{ 
			cv::Point{0,          size.height}, 
			cv::Point{0,          0}, 
			cv::Point{size.width, 0}, 
			cv::Point{size.width, size.height} 
		};
		auto transform_matrix = cv::getPerspectiveTransform(vertex, target_points);
		auto text_image = cv::Mat{};
		cv::warpPerspective(image, text_image, transform_matrix, size);
		text_images.emplace_back(text_image);
	}

	return text_images;
}
void crnn_ctc() {
	// ファイルを読み込む。
	const auto IMAGE_PATH = std::string{"C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.6\\db\\text.jpg"};
	const auto is_good_image = load_image(IMAGE_PATH);
	if (!is_good_image) {
		std::cout << "invalid file path\n";
		return;
	}
	const cv::Mat image = is_good_image.value();

	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH = std::string{ "C:\\data\\opencv_book\\7.6\\db\\DB_IC15_resnet50.onnx" };
	if (!fs::exists(WEIGHTS_PATH)) {
		std::cout << "invalid model path\n";
		return;
	}

	// 検出器を作る。
	auto detector = TextDetector{ WEIGHTS_PATH };


	// テキスト認識器を読み込む
	// 重みファイルパスを確認する。
	const auto WEIGHTS_PATH_2 = std::string{ "C:\\data\\opencv_book\\7.6\\crnn-ctc\\crnn_cs.onnx" };
	if (!fs::exists(WEIGHTS_PATH_2)) {
		std::cout << "invalid model path\n";
		return;
	}
	// 語彙リストパスを確認する。
	const auto VOC = std::string{ "C:\\projects\\opencv_book\\opencv_dl_book\\ch7\\7.6\\crnn-ctc\\alphabet_94.txt" };
	if (!fs::exists(VOC)) {
		std::cout << "invalid alphabet path\n";
		return;
	} 
	auto recognizer = TextRecognizer(WEIGHTS_PATH_2, VOC);


	// テキストを検出する。
	auto text_info = detector.detect_vertices(image);
	if (!text_info) {
		std::cout << "invalid detection\n";
		return;
	}
	const auto& vertices = text_info.value().vertices_;
	
	// テキスト領域の画像を切り出す。
	auto text_images = get_text_images(image, vertices);


    // テキストを認識する
	auto texts = std::vector<std::string>{};
	texts.reserve(text_images.size());
	for (const auto& text_image : text_images) {
		auto text = recognizer.recognize(text_image);
		if (text) {
			texts.emplace_back(text.value());
		}
	}

	// テキスト検出の結果を描画する
	for (const auto& vertex : vertices) {
		auto close = true;
		auto color = cv::Scalar{ 0, 255, 0 };
		auto thickness = 2;
		cv::polylines(image, vertex, close, color, thickness, cv::LINE_AA);
	}

    // テキスト認識の結果を描画する
	for (const auto& p : boost::combine(texts, vertices)) {
		const std::string& text = boost::get<0>(p);
		const std::vector<cv::Point>& vertex = boost::get<1>(p);

		auto position = vertex[1] - cv::Point(0, 10);
		auto font = cv::FONT_HERSHEY_SIMPLEX;
		auto scale = 1.0;
		auto color = cv::Scalar{ 0, 0, 255 };
		auto thickness = 2;
		cv::putText(image, text, position, font, scale, color, thickness, cv::LINE_AA);

		// OpenCVのcv2.putText()では中国語（漢字）は描画できないので標準出力に表示する
		std::cout << text << std::endl;
	}

	cv::imshow("image", image);
	cv::waitKey(0);
}