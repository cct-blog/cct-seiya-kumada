#include <opencv2/opencv.hpp>
#include <iostream>
#include <format>

void fill_with_black(cv::Mat& roi);

int main(int argc, char* argv[]) {
	const auto IMAGE_PATH = std::string{"C:/data/cct_blog/opencv_koneta/Parrots.bmp"};
	auto image = cv::imread(IMAGE_PATH);
	const auto rows = image.rows;
	const auto cols = image.cols;
	std::cout << std::format("rows:{}, cols:{}", rows,  cols) << std::endl;
	
	auto x = cols / 2;
	auto y = rows / 2;
	auto width = 50;
	auto height = 50;
	auto rect = cv::Rect(x, y, width, height);

	// ROIの抽出（コピーではなく参照である）
	auto roi = image(rect);// .clone();

	// ROIを真っ黒にする。
	fill_with_black(roi);

	cv::imshow("image", image);
	cv::imwrite("C:/data/cct_blog/opencv_koneta/Parrots_roi.jpg", image);
	cv::waitKey(0);
	return 0;
}

void fill_with_black(cv::Mat& roi) {
	const auto rows = roi.rows;
	const auto cols = roi.cols;

	for (auto j = 0; j < rows; ++j) {
		cv::Vec3b* row = roi.ptr<cv::Vec3b>(j); // j行目の先頭ポインタを取得
		for (auto i = 0; i < cols; ++i) {
			const cv::Vec3b& pixel = row[i]; // j行目、i列目の画素を取得

			// ピクセルを更新
			row[i] = cv::Vec3b(0, 0, 0);
		}
	}
}
