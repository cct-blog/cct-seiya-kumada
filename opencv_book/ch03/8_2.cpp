#include "8_2.h"
#include "CropLayer.h"

void dnn_custom_layer() {

	// カスタムレイヤの登録
	// これはネットワークを構成する要素を登録している。
	cv::dnn::LayerFactory::registerLayer("Crop", CropLayer::create);
}