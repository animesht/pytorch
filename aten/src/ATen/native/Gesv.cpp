#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/Gesv.h"

#include "TH.h"  // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dgesv_(
    int* n, int* nrhs, double* a, int* lda,
    int *ipiv, double* b, int* ldb, int* info);
extern "C" void sgesv_(
    int* n, int* nrhs, float* a, int* lda,
    int* ipiv, float* b, int* ldb, int* info);
#endif

namespace at { namespace native {

template<class scalar_t>
void lapackGesv(
    int n, int nrhs, scalar_t* a, int lda, int* ipiv,
    scalar_t* b, int ldb, int* info) {
  AT_ERROR("gesv only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGesv<float>(
    int n, int nrhs, float* a, int lda, int* ipiv,
    float* b, int ldb, int* info) {
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGesv<double>(
    int n, int nrhs, double* a, int lda, int* ipiv,
    double* b, int ldb, int* info) {
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}
#endif

template <typename scalar_t>
static void applyGesv(Tensor& b, Tensor& A, std::vector<int64_t> infos) {
#ifndef USE_LAPACK
  AT_ERROR("gesv: LAPACK library not found in compilation");
#endif
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  auto ipiv = at::empty({n}, b.type().toScalarType(kInt));

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    lapackGesv<scalar_t>(n, nrhs, A_working_ptr, n, ipiv.data<int>(),
        b_working_ptr, n, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

static std::tuple<Tensor,Tensor> _gesv_single_helper(const Tensor& self, const Tensor& A) {
#ifndef USE_LAPACK
  AT_ERROR("gesv: LAPACK library not found in compilation");
#endif
  int64_t bx = self.size(0);
  int64_t by = (self.dim() == 1) ? 1 : self.size(1);
  int info = 0;

  auto A_ = A.t().clone();
  auto b_ = self.view({bx, by}).t().clone();

  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    auto A_ptr = A_.data<scalar_t>();
    auto b_ptr = b_.data<scalar_t>();
    auto ipiv = at::empty({bx}, b_.type().toScalarType(kInt));

    lapackGesv<scalar_t>(bx, by, A_ptr, bx, ipiv.data<int>(), b_ptr, bx, &info);
  });

  checkErrors({info});
  return std::tuple<Tensor, Tensor>(b_.t(), A_.t());
}

std::tuple<Tensor,Tensor> _gesv_helper_cpu(const Tensor& self, const Tensor& A) {
  std::vector<int64_t> infos(batchCount(A), 0);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  auto b_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    applyGesv<scalar_t>(b_working_copy, A_working_copy, infos);
  });
  checkErrors(infos);
  return std::tuple<Tensor,Tensor>(b_working_copy, A_working_copy);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor,Tensor> gesv(const Tensor& self, const Tensor& A) {
  bool batched = !(self.dim() <= 2 && A.dim() <= 2);
  checkInputs(self, A, batched);

  if (!batched) {
    return _gesv_single_helper(self, A);
  }

  // broadcast the batch dimensions of self and A.
  IntList self_batch_sizes(self.sizes().data(), self.ndimension() - 2);
  IntList A_batch_sizes(A.sizes().data(), A.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion =
      infer_size(self_batch_sizes, A_batch_sizes);

  std::vector<int64_t> self_expand_size({expand_batch_portion});
  self_expand_size.insert(self_expand_size.end(),
      { self.size(-2), self.size(-1) });

  std::vector<int64_t> A_expand_size({expand_batch_portion});
  A_expand_size.insert(A_expand_size.end(),
      { A.size(-2), A.size(-1) });

  Tensor self_broadcasted  = self.expand(self_expand_size);
  Tensor A_broadcasted = A.expand(A_expand_size);
  return self.type()._gesv_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&,Tensor&> gesv_out(
    Tensor& solution, Tensor& lu, const Tensor& self, const Tensor& A) {
  if (self.dim() > 2 || A.dim() > 2) {
    AT_ERROR("torch.gesv() with the `out` keyword does not support batching. "
                  "b.dim() (%lld) and A.dim() (%lld) must both be 2.",
                  (long long)self.dim(), (long long)A.dim());
  }
  return at::_th_gesv_single_out(solution, lu, self, A);
}

}}  // namespace at::native
