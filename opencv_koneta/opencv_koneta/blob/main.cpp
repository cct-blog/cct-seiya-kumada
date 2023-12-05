#include <opencv2/opencv.hpp>
#include <boost/format.hpp>

int main(int argc, char* argv[]) {
	// blobに設定したい画像
	const auto IMAGE_PATH = std::string{"C:/data/cct_blog/opencv_koneta/icelandlake2540.jpg"};
		
    // カラー画像を読み込む
    cv::Mat originalImage = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    if (originalImage.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }

    std::cout << boost::format("rows: %1%") % originalImage.rows << std::endl;
    std::cout << boost::format("cols: %1%") % originalImage.cols << std::endl;
    std::cout << boost::format("channels: %1%") % originalImage.channels() << std::endl;


    // blobFromImage を使って画像から blob を作成
    // パラメータは適宜調整する。
    double scalefactor = 1.0; // スケールファクター
    cv::Size size = cv::Size(originalImage.cols, originalImage.rows); // 入力サイズ
    cv::Scalar mean = cv::Scalar(0, 0, 0); // 平均減算の値
    bool swapRB = true; // OpenCV は BGR 形式で読み込むため、通常は swapRB = true に設定

    auto originalImages = std::vector<cv::Mat>{ originalImage, originalImage.clone() };
    auto inputBlob = cv::dnn::blobFromImages(originalImages, scalefactor, size, mean, swapRB, false);

    std::cout << boost::format("dim: %1%") % inputBlob.dims << std::endl;
    std::cout << boost::format("batch size: %1%") % inputBlob.size[0] << std::endl;
    std::cout << boost::format("channels: %1%") % inputBlob.size[1] << std::endl;
    std::cout << boost::format("height: %1%") % inputBlob.size[2] << std::endl;
    std::cout << boost::format("width: %1%") % inputBlob.size[3] << std::endl;

    std::cout << boost::format("rows: %1%") % inputBlob.rows << std::endl; // -1が出る。無効な値
    std::cout << boost::format("cols: %1%") % inputBlob.cols << std::endl; // -1が出る。無効な値
    std::cout << boost::format("channels: %1%") % inputBlob.channels() << std::endl; // 1が出る。無効な値

    auto height = inputBlob.size[2];
    auto width = inputBlob.size[3];
   	auto redImage = cv::Mat{ height, width, CV_32F, inputBlob.ptr<float>(0, 0) };
   	auto greenImage = cv::Mat{ height, width, CV_32F, inputBlob.ptr<float>(0, 1) };
   	auto blueImage = cv::Mat{ height, width, CV_32F, inputBlob.ptr<float>(0, 2) };

   	// 画像を正しい形式に変換 (CV_32F -> CV_8U)
    redImage.convertTo(redImage, CV_8U, 1.0); // スケーリング
    greenImage.convertTo(greenImage, CV_8U, 1.0); // スケーリング
    blueImage.convertTo(blueImage, CV_8U, 1.0); // スケーリング
    
    // カラー画像へ
    auto channels = std::vector<cv::Mat>{ blueImage, greenImage, redImage };
    auto colorImage = cv::Mat{};
    cv::merge(channels, colorImage);

    // 画像を表示
    cv::imshow("Input Blob Image", colorImage);
    cv::waitKey(0);
	return 0;
}