#include "8_2.h"
#include "CropLayer.h"

void dnn_custom_layer() {

	// �J�X�^�����C���̓o�^
	// ����̓l�b�g���[�N���\������v�f��o�^���Ă���B
	cv::dnn::LayerFactory::registerLayer("Crop", CropLayer::create);
}