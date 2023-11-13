#include "7_6.h"
#include <string>
#include "util.h"
#include <iostream>
#include "TextDetector.h"
#include "TextRecognizer.h"
#include <filesystem>
#include <boost/range/combine.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <format>

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

auto debug_print_vertex(const std::vector<cv::Point>& vertex) {
	for (const auto& v : vertex) {
		std::cout << v << std::endl;
	}
	std::cout << "---\n";
}

auto convert_to_point2f(const std::vector<cv::Point>& vertex) {
	auto results = std::vector<cv::Point2f>{};
	results.reserve(vertex.size());
	for (const auto& p : vertex) {
		results.emplace_back(p.x, p.y);
	}
	return results;
}

auto get_text_images(
	const cv::Mat& image, 
	const std::vector<std::vector<cv::Point>>& vertices) 
-> std::vector<cv::Mat> {
	auto text_images = std::vector<cv::Mat>{};
	auto size = cv::Size2f{ 100.0f, 32.0f }; // w, h
		
	text_images.reserve(vertices.size());
	for (const auto& vertex : vertices) {
		//std::cout << std::size(vertex) << std::endl;
		auto src_points = convert_to_point2f(vertex);
		auto dst_points = std::vector<cv::Point2f>{ 
			cv::Point2f{0.0f,          size.height}, 
			cv::Point2f{0.0f,          0.0f}, 
			cv::Point2f{size.width, 0.0f}, 
			cv::Point2f{size.width, size.height} 
		};
		auto transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
		auto text_image = cv::Mat{};
		cv::warpPerspective(image, text_image, transform_matrix, size);
		text_images.emplace_back(text_image);
	}

	return text_images;
}

auto save_text_images(const std::vector<cv::Mat>& text_images) {
	const auto ROOT_DIR_PATH = fs::path{ "C:\\data\\opencv_book\\7.6_outputs" };
	for (const auto& p :text_images | boost::adaptors::indexed(0)) {
		auto i = p.index();
		auto v = p.value();
		auto path = ROOT_DIR_PATH / std::format("{}.jpg", i);
		cv::imwrite(path.string(), v);
	}
}

auto print_texts(const std::vector<std::string>& texts) {
	for (const auto& text : texts) {
		std::cout << text << std::endl;
	}
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
	//save_text_images(text_images);

    // テキストを認識する
	auto texts = std::vector<std::string>{};
	texts.reserve(text_images.size());
	for (const auto& text_image : text_images) {
		auto text = recognizer.recognize(text_image);
		if (text) {
			texts.emplace_back(text.value());
		}
	}
	//print_texts(texts);

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
	const auto output_path = std::string{ "C:\\data\\opencv_book\\7.6_outputs\\result.jpg" };
	cv::imwrite(output_path, image);
	cv::waitKey(0);
}