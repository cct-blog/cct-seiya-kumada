#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
namespace chrono = std::chrono;

void method_at(const cv::Mat& image);
void method_pointer(const cv::Mat& image);
void method_iterator(const cv::Mat& image);
typedef void Method(const cv::Mat&);

void repeat(int num, Method method, const cv::Mat& image);

int main(int argc, char* argv[]) {
	const auto WIDTH = 2000;
	const auto HEIGHT = 2000;
	const auto image = cv::Mat{ HEIGHT, WIDTH, CV_8UC3, cv::Scalar{0, 0, 0} };
	const auto NUM = 100;

	{
		auto start = chrono::high_resolution_clock::now();
		repeat(NUM, method_at, image);
		auto end = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "average time(at): " << duration / static_cast<float>(NUM) << "[ms per image]" << std::endl;
	}

	{
		auto start = chrono::high_resolution_clock::now();
		repeat(NUM, method_pointer, image);
		auto end = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "average time(pointer): " << duration / static_cast<float>(NUM) << "[ms per image]" << std::endl;
	}

	{
		auto start = chrono::high_resolution_clock::now();
		repeat(NUM, method_iterator, image);
		auto end = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "average time(iterator): " << duration / static_cast<float>(NUM) << "[ms per image]" << std::endl;
	}
	return 0;
}

// �J��Ԃ����s���B
void repeat(int num, Method method, const cv::Mat& image) {
	for (auto i = 0; i < num; ++i) {
		method(image);
	}
}

// at��p��������
void method_at(const cv::Mat& image) {
	const auto rows = image.rows;
	const auto cols = image.cols;

	// �o�͉摜���쐬
	auto target = cv::Mat{ rows, cols, CV_8UC3, cv::Scalar{0, 0, 0} };

	for (auto j = 0; j < rows; ++j) {
		for (auto i = 0; i < cols; ++i) {
			// �s�N�Z���ɃA�N�Z�X
			cv::Vec3b pixel = image.at<cv::Vec3b>(j, i); // j�s�ځAi��ڂ̉�f���擾
			
			// �F�������擾
			uchar blue = pixel[0];
			uchar green = pixel[1];
			uchar red = pixel[2];
			
			// �F������ύX
			uchar new_blue = blue / 2;
			uchar new_green = green / 2;
			uchar new_red = red / 2;

			// �s�N�Z�����X�V
			target.at<cv::Vec3b>(j, i) = cv::Vec3b(new_blue, new_green, new_red);
		}
	}
}

// pointer��p��������
void method_pointer(const cv::Mat& image) {
	const auto rows = image.rows;
	const auto cols = image.cols;
	auto target = cv::Mat{ rows, cols, CV_8UC3, cv::Scalar{0, 0, 0} };

	for (auto j = 0; j < rows; ++j) {
		const cv::Vec3b* row = image.ptr<cv::Vec3b>(j); // j�s�ڂ̐擪�|�C���^���擾
		cv::Vec3b* target_row = target.ptr<cv::Vec3b>(j); // j�s�ڂ̐擪�|�C���^���擾
		for (auto i = 0; i < cols; ++i) {
			const cv::Vec3b& pixel = row[i]; // j�s�ځAi��ڂ̉�f���擾

			// �F�������擾
			uchar blue = pixel[0];
			uchar green = pixel[1];
			uchar red = pixel[2];
			
			// �F������ύX
			uchar new_blue = blue / 2;
			uchar new_green = green / 2;
			uchar new_red = red / 2;

			// �s�N�Z�����X�V
			target_row[i] = cv::Vec3b(new_blue, new_green, new_red);
		}
	}

}

// iterator��p��������
void method_iterator(const cv::Mat& image) {
	const auto rows = image.rows;
	const auto cols = image.cols;
	auto target = cv::Mat{ rows, cols, CV_8UC3, cv::Scalar{0, 0, 0} };

	auto src_it = image.begin<cv::Vec3b>();
	auto dst_it = target.begin<cv::Vec3b>();
	auto src_end = image.end<cv::Vec3b>();

	while (src_it != src_end) {
        const cv::Vec3b& pixel = *src_it;

		// �F�������擾
		uchar blue = pixel[0];
		uchar green = pixel[1];
		uchar red = pixel[2];
		
		// �F������ύX
		uchar new_blue = blue / 2;
		uchar new_green = green / 2;
		uchar new_red = red / 2;

		// �s�N�Z�����X�V
		*dst_it = cv::Vec3b(new_blue, new_green, new_red);

		++src_it;
		++dst_it;
	}
}

