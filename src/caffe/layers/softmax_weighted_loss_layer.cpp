#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace{
  const float MIN_VALUE = 1e-15f;

  std::string ints_to_string(const std::vector<int>& vec){
    std::stringstream result;
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(result, " "));
    return result.str();
  }
}

namespace caffe {

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  auto& softmax_param = this->layer_param_;
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  auto& weighted_loss_param = this->layer_param_.weighted_loss_param();
  has_ignore_label_ = weighted_loss_param.has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = weighted_loss_param.ignore_label();
  }

  class_labels_.clear();
  int pixel_label_size = weighted_loss_param.pixel_label_size();
  class_labels_.resize(pixel_label_size);
  for(int i = 0; i < weighted_loss_param.pixel_label_size(); i++){
    class_labels_[i] = weighted_loss_param.pixel_label(i);
  }
  LOG(INFO) << " Ground truth labels: " << ints_to_string(class_labels_);

  const vector<int>& labelShape = bottom[1]->shape();
  pixel_weights_.Reshape(labelShape);
}

template <typename Dtype>
void  SoftmaxWithWeightedLossLayer<Dtype>::recalculate_pixel_weights(const Blob<Dtype>& label){
  // Zero out the weights blob
  caffe_set<Dtype>(pixel_weights_.count(), static_cast<Dtype>(0), pixel_weights_.mutable_cpu_data());

  // Count up the number of times each label value is present in the image
  // and divide 1 by the count. If a label is rare it should be weighted higher within the loss function

  // Analyze each element in the batch separately and build a lookup table for each batch element
  vector<std::map<int, Dtype> > label_weight_luts;
  for (int i = 0; i < outer_num_; ++i) {
    std::map<int, Dtype> label_weight_lut;
    for(vector<int>::const_iterator j = class_labels_.begin(); j != class_labels_.end(); ++j){
      // Find all instances of this label value within this element in the batch
      const Dtype label_value = static_cast<Dtype>(*j);
      const Dtype* begin_ptr = label.cpu_data() + (i*inner_num_);
      const Dtype* end_ptr = label.cpu_data() + (i+1)*inner_num_;

      int label_count = std::count(begin_ptr, end_ptr, label_value);
      if(label_count > 0){
        label_weight_lut[*j] = static_cast<Dtype>(1.0 / label_count);
      }
      else{
        label_weight_lut[*j] = static_cast<Dtype>(1.0);
      }
    }
    label_weight_luts.push_back(label_weight_lut);
  }
  const Dtype* label_ptr = label.cpu_data();
  Dtype* weight_ptr = pixel_weights_.mutable_cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      // Label at this pixel
      const int label_value = static_cast<int>(label_ptr[i * inner_num_ + j]);
      // Weight for this label
      weight_ptr[i * inner_num_ + j] = label_weight_luts[i][label_value];
    }
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  pixel_weights_.Reshape(bottom[1]->shape());
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  recalculate_pixel_weights(*bottom[1]);
  const Dtype* weight = pixel_weights_.cpu_data();
  const int dim = prob_.count() / outer_num_;
  Dtype loss = static_cast<Dtype>(0);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));

      const Dtype& prob_value = prob_data[i * dim + label_value * inner_num_ + j];
      loss -= weight[i * inner_num_ + j] * log(std::max(prob_value, static_cast<Dtype>(MIN_VALUE)));
    }
  }

  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* weight = pixel_weights_.cpu_data();
    int dim = prob_.count() / outer_num_;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const Dtype weight_value = static_cast<Dtype>(weight[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            if (c == label_value) {
              bottom_diff[i * dim + c * inner_num_ + j] -=1;
            }
            bottom_diff[i * dim + c * inner_num_ + j] *= weight_value;
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithWeightedLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithWeightedLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithWeightedLoss);

}  // namespace caffe
