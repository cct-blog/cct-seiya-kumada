#include <opencv2/opencv.hpp>
#include <format>

int main(int argc, char* argv[]) {
	// blob�ɐݒ肵�����摜
	const auto IMAGE_PATH = std::string{"C:/data/cct_blog/opencv_koneta/icelandlake2540.jpg"};
		
    // �J���[�摜��ǂݍ���
    cv::Mat originalImage = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    if (originalImage.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }

    std::cout << std::format("rows: {}", originalImage.rows) << std::endl;
    std::cout << std::format("cols: {}", originalImage.cols) << std::endl;
    std::cout << std::format("channels: {}", originalImage.channels()) << std::endl;


    // blobFromImage ���g���ĉ摜���� blob ���쐬
    // �p�����[�^�͓K�X��������B
    double scalefactor = 1.0; // �X�P�[���t�@�N�^�[
    cv::Size size = cv::Size(originalImage.cols, originalImage.rows); // ���̓T�C�Y
    cv::Scalar mean = cv::Scalar(0, 0, 0); // ���ό��Z�̒l
    bool swapRB = true; // OpenCV �� BGR �`���œǂݍ��ނ��߁A�ʏ�� swapRB = true �ɐݒ�

    auto originalImages = std::vector<cv::Mat>{ originalImage, originalImage.clone() };
    auto inputBlob = cv::dnn::blobFromImages(originalImages, scalefactor, size, mean, swapRB, false);

    std::cout << std::format("dim: {}", inputBlob.dims) << std::endl;
    std::cout << std::format("batch size: {}", inputBlob.size[0]) << std::endl;
    std::cout << std::format("channels: {}", inputBlob.size[1]) << std::endl;
    std::cout << std::format("height: {}", inputBlob.size[2]) << std::endl;
    std::cout << std::format("width: {}", inputBlob.size[3]) << std::endl;

    std::cout << std::format("rows: {}", inputBlob.rows) << std::endl; // -1���o��B�����Ȓl
    std::cout << std::format("cols: {}", inputBlob.cols) << std::endl; // -1���o��B�����Ȓl
    std::cout << std::format("channels: {}", inputBlob.channels()) << std::endl; // 1���o��B�����Ȓl

    auto height = inputBlob.size[2];
    auto width = inputBlob.size[3];
   	auto redImage = cv::Mat{ height, width, CV_32F, inputBlob.ptr<float>(0, 0) };
   	auto greenImage = cv::Mat{ height, width, CV_32F, inputBlob.ptr<float>(0, 1) };
   	auto blueImage = cv::Mat{ height, width, CV_32F, inputBlob.ptr<float>(0, 2) };

   	// �摜�𐳂����`���ɕϊ� (CV_32F -> CV_8U)
    redImage.convertTo(redImage, CV_8U, 1.0); // �X�P�[�����O
    greenImage.convertTo(greenImage, CV_8U, 1.0); // �X�P�[�����O
    blueImage.convertTo(blueImage, CV_8U, 1.0); // �X�P�[�����O
    
    // �J���[�摜��
    auto channels = std::vector<cv::Mat>{ blueImage, greenImage, redImage };
    auto colorImage = cv::Mat{};
    cv::merge(channels, colorImage);

    // �摜��\��
    cv::imshow("Input Blob Image", colorImage);
    cv::waitKey(0);
	return 0;
}