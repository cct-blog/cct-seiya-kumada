#include "8_2.h"
#include "CropLayer.h"
#include <filesystem>
namespace fs = std::filesystem;

void dnn_custom_layer() {

	// �J�X�^�����C���̓o�^
	// ����̓l�b�g���[�N���\������v�f��o�^���Ă���B
	cv::dnn::LayerFactory::registerLayer("Crop", CropLayer::create);

	// �\�����f����ǂݍ��ށB
	const auto weights_path = std::string{ "C:/data/opencv_book/8.2/hed/hed_pretrained_bsds.caffemodel" };
	const auto config_path = std::string{ "C:/data/opencv_book/8.2/hed/examples/hed/deploy.prototxt" };
	auto net = cv::dnn::readNet(weights_path, config_path);

	// �摜��ǂݍ��ށB
	const auto FILE_PATH = std::string{ "C:/projects/opencv_book/opencv_dl_book/ch8/8.2/yorkie.png" };
	auto image = cv::imread(FILE_PATH, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "invalid file\n";
		return;
	}

	
	//std::cout << "image size " << image.size() << std::endl; // 480x320
	//std::cout << "image rows " << image.rows << std::endl; // 320
	//std::cout << "image cols " << image.cols << std::endl; // 480
	//std::cout << "image channels " << image.channels() << std::endl; // 3


	// ���̓p�����[�^�[����摜���u���u�ɕϊ�����
	auto blob = cv::dnn::blobFromImage(
		image, 
		1.0, // scalefactor
		cv::Size{}, // size of output image
		cv::Scalar{ 104.00698793, 116.66876762, 122.67891434 }, 
		false, // swap
		false  // crop
	);
	//std::cout << "blob dim " << blob.dims << std::endl; // 4
	//std::cout << "blob batch size " << blob.size[0] << std::endl; // 1
	//std::cout << "blob c " << blob.size[1] << std::endl; // 3
	//std::cout << "blob h(rows) " << blob.size[2] << std::endl; // 320
	//std::cout << "blob w(cols) " << blob.size[3] << std::endl; // 480

	// �l�b�g���[�N�̓��̓��C���Ƀu���u��ݒ肷��
	net.setInput(blob);

	// �l�b�g���[�N�����`�����Đ��_���ʂ��擾����  
	auto out = net.forward();
	//std::cout << "dim " << out.dims << std::endl; // 4
	//std::cout << "batch size " << out.size[0] << std::endl; // 1
	//std::cout << "c " << out.size[1] << std::endl; // 3
	auto height = out.size[2];
	auto width = out.size[3];

	auto grayImage = cv::Mat{ height, width, CV_32F, out.ptr<float>(0, 0) };

	// �摜�𐳂����`���ɕϊ� (CV_32F -> CV_8U)
    grayImage.convertTo(grayImage, CV_8U, 255.0); // �X�P�[�����O

	cv::imshow("result", grayImage);
	//auto output_path = "C:\\data\\opencv_book\\face_detetion_result.jpg";
	//cv::imwrite(output_path, image);
	cv::waitKey(0);



}