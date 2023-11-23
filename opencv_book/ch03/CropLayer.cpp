#include "CropLayer.h"

CropLayer::CropLayer(const cv::dnn::LayerParams& params)
	: cv::dnn::Layer(params)
	, xstart_{ 0 }
	, xend_{ 0 }
	, ystart_{ 0 }
	, yend_{ 0 }{}

cv::Ptr<cv::dnn::Layer> CropLayer::create(cv::dnn::LayerParams& params) {
	return cv::makePtr<CropLayer>(params);
}

void CropLayer::forward(
	std::vector<cv::Mat*>&	inputs,
	std::vector<cv::Mat>&	outputs,
	std::vector<cv::Mat>&	internals) {

	auto r = inputs[0][0](cv::Rect(xstart_, ystart_, xend_ - xstart_, yend_ - ystart_)).clone();
	outputs.emplace_back(r);
}

bool CropLayer::getMemoryShapes(
	const std::vector<cv::dnn::MatShape>&	inputs,
	const int								requiredOutputs, 
	std::vector<cv::dnn::MatShape>&			outputs,
	std::vector<cv::dnn::MatShape>&			internals) const {

	const auto& inputShape = inputs[0];
	const auto& targetShape = inputs[1];
	
	// MatShape: 0:batch, 1:channel, 2:height, 3:width
	const auto batchSize = inputShape[0];
	const auto numChannels = inputShape[1];
	const auto inputHeight = inputShape[2];
	const auto inputWidth = inputShape[3];

	const auto targetHeight = targetShape[2];
	const auto targetWidth = targetShape[3];

	xstart_ = (inputWidth - targetWidth) / 2;
	ystart_ = (inputHeight - targetHeight) / 2;
	xend_ = xstart_ + targetWidth;
	yend_ = ystart_ + targetHeight;

	outputs.clear();
	auto v = cv::dnn::MatShape{ batchSize, numChannels, targetHeight, targetWidth };
	outputs.emplace_back(v);
	return true;
}

