#include "ATen/ATen.h"

namespace at { namespace native {

static inline void checkInputs(const Tensor& self, const Tensor& A, bool batched) {

  if (A.size(-1) != A.size(-2)) {
    if (batched) {
      AT_ERROR("A must be batches of square matrices, "
          "but they are %lld by %lld matrices",
          (long long)A.size(-1), (long long)A.size(-2));
    } else {
      AT_ERROR("A must be a square matrix, but is %lld by %lld",
          (long long)A.size(-1), (long long)A.size(-2));
    }
  }

  if (batched && A.size(-1) != self.size(-2)) {
    AT_ERROR("Incompatible matrix sizes for matmul: each A "
        "matrix is %llu by %lld but each b matrix is %lld by %lld.",
        (long long)A.size(-1), (long long)A.size(-1),
        (long long)self.size(-2), (long long)self.size(-1));
  } else if (A.size(0) != self.size(0)) {
    AT_ERROR("A,B size incompatible - A has %ld "
        "rows, B has %ld cols", A.size(0), self.size(0));
  }
}

static inline void checkErrors(std::vector<int64_t> infos) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR("gesv: For batch %lld: Argument %lld has illegal value",
          (long long)i, -info);
    } else if (info > 0) {
      AT_ERROR("gesv: For batch %lld: U(%lld,%lld) is zero, singular U.",
          (long long)i, info, info);
    }
  }
}

}}  // namespace at::native
