#include "8_2.h"
#include "CropLayer.h"

void dnn_custom_layer() {

	// �J�X�^�����C���̓o�^
	// ����̓l�b�g���[�N���\������v�f��o�^���Ă���B
	cv::dnn::LayerFactory::registerLayer("Crop", CropLayer::create);

	// �\�����f����ǂݍ��ށB
	const auto weights_path = std::string{"C:/data/opencv_book/8.2/hed/hed_pretrained_bsds.caffemodel"};
	const auto config_path = std::string{ "C:/data/opencv_book/8.2/hed/examples/hed/deploy.prototxt"};
	auto net = cv::dnn::readNet(weights_path, config_path);
	
	std::cout << "hoge" << std::endl;

}