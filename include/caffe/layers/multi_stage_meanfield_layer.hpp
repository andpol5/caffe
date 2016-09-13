#ifndef CAFFE_MULTISTAGEMEANFIELD_LAYER_HPP_
#define CAFFE_MULTISTAGEMEANFIELD_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/meanfield_iteration.hpp"
#include "caffe/util/modified_permutohedral.hpp"


namespace caffe {
template <typename Dtype>
class MultiStageMeanfieldLayer : public Layer<Dtype> {

public:
  explicit MultiStageMeanfieldLayer(const LayerParameter& param) : Layer<Dtype>(param),
    init_cpu_(false), init_gpu_(false) {}
  virtual ~MultiStageMeanfieldLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline const char* type() const {
      return "MultiStageMeanfield";
  }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  void compute_spatial_kernel(float* const output_kernel);
  void compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel);
  void init_param_blobs(const MultiStageMeanfieldParameter& meanfield_param);
  void init_spatial_lattice();
  void init_bilateral_buffers();

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Dtype theta_alpha_;
  Dtype theta_beta_;
  Dtype theta_gamma_;
  int num_iterations_;

  /*float**/ Dtype* norm_feed_;  // The permutehedral lattice is not templated.
  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  vector<Blob<Dtype>*> split_layer_bottom_vec_;
  vector<Blob<Dtype>*> split_layer_top_vec_;

  // Blobs owned by this instance of the class.
  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;

  vector<shared_ptr<MeanfieldIteration<Dtype> > > meanfield_iterations_;

  shared_ptr<SplitLayer<Dtype> > split_layer_;

  // FIXME: Not very efficient. Since the same softmax normalisation (with the same input data)
  // will be done in the first mean-field iteration
  Blob<Dtype> unary_prob_;

  // Permutohedral lattice
  shared_ptr<ModifiedPermutohedral> spatial_lattice_;
  /*Dtype* */ float* bilateral_kernel_buffer_;   // Permutohedral lattice is not templated
  vector<shared_ptr<ModifiedPermutohedral> > bilateral_lattices_;

  /* GPU/CPU stuff */
  bool init_cpu_;
  bool init_gpu_;
};
} //namespace caffe

#endif
