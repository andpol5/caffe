#ifndef CAFFE_IMAGE_LABEL_DATA_LAYER_H
#define CAFFE_IMAGE_LABEL_DATA_LAYER_H

#include <random>
#include <vector>

#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template<typename Dtype>
class ImageLabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageLabelDataLayer(const LayerParameter &param);
  virtual ~ImageLabelDataLayer();

  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }

  virtual inline const char *type() const { return "ImageLabelData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;

  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;

  int batch_size_;
  int crop_size_;

  std::string image_dir_;
  std::string label_dir_;
  vector<int> image_shape_;
  vector<int> label_shape_;

  Blob<Dtype> transformed_data_;
  Blob<Dtype> transformed_label_;

  std::mt19937 *rng_;
};

} // namspace caffe

#endif //CAFFE_IMAGE_LABEL_DATA_LAYER_H
