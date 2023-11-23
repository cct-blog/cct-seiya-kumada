#include "8_2.h"
#include "CropLayer.h"

void dnn_custom_layer() {

	// カスタムレイヤの登録
	// これはネットワークを構成する要素を登録している。
	cv::dnn::LayerFactory::registerLayer("Crop", CropLayer::create);

	// 予測モデルを読み込む。
	const auto weights_path = std::string{"C:/data/opencv_book/8.2/hed/hed_pretrained_bsds.caffemodel"};
	const auto config_path = std::string{ "C:/data/opencv_book/8.2/hed/examples/hed/deploy.prototxt"};
	auto net = cv::dnn::readNet(weights_path, config_path);
	
	std::cout << "hoge" << std::endl;

}