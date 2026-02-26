#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct ColMat {
  int rows = 0;
  int cols = 0;
  std::vector<double> v;  // column-major

  ColMat() = default;
  ColMat(int r, int c) : rows(r), cols(c), v(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0) {}

  void resize(int r, int c) {
    rows = r;
    cols = c;
    v.assign(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0);
  }

  inline double& operator()(int r, int c) { return v[static_cast<size_t>(c) * static_cast<size_t>(rows) + r]; }
  inline const double& operator()(int r, int c) const {
    return v[static_cast<size_t>(c) * static_cast<size_t>(rows) + r];
  }

  inline double* col_ptr(int c) { return v.data() + static_cast<size_t>(c) * static_cast<size_t>(rows); }
  inline const double* col_ptr(int c) const { return v.data() + static_cast<size_t>(c) * static_cast<size_t>(rows); }

  void set_zero() { std::fill(v.begin(), v.end(), 0.0); }
};

static inline double threshold_l1(double x, double thr) {
  if (x > thr) return x - thr;
  if (x < -thr) return x + thr;
  return 0.0;
}

static ColMat to_colmat(py::array_t<double, py::array::c_style | py::array::forcecast> arr, const char* name) {
  auto a = arr.unchecked<2>();
  const py::ssize_t r = a.shape(0);
  const py::ssize_t c = a.shape(1);
  if (r <= 0 || c <= 0) {
    throw std::runtime_error(std::string(name) + " must be non-empty.");
  }
  ColMat out(static_cast<int>(r), static_cast<int>(c));
  for (int i = 0; i < out.rows; ++i) {
    for (int j = 0; j < out.cols; ++j) {
      out(i, j) = a(i, j);
    }
  }
  return out;
}

static py::array_t<double> to_numpy_2d_double(const ColMat& m) {
  auto out = py::array_t<double>({m.rows, m.cols});
  auto o = out.mutable_unchecked<2>();
  for (int i = 0; i < m.rows; ++i) {
    for (int j = 0; j < m.cols; ++j) {
      o(i, j) = m(i, j);
    }
  }
  return out;
}

static ColMat submatrix(const ColMat& m, const std::vector<int>& idx) {
  const int q = static_cast<int>(idx.size());
  ColMat out(q, q);
  for (int i = 0; i < q; ++i) {
    for (int j = 0; j < q; ++j) {
      out(i, j) = m(idx[i], idx[j]);
    }
  }
  return out;
}

static double det_gaussian_elimination(const ColMat& m) {
  if (m.rows != m.cols) return 0.0;
  const int n = m.rows;
  std::vector<double> a(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[static_cast<size_t>(i) * static_cast<size_t>(n) + j] = m(i, j);
    }
  }

  double det_sign = 1.0;
  double det_val = 1.0;
  constexpr double eps = 1e-15;
  for (int k = 0; k < n; ++k) {
    int pivot = k;
    double max_abs = std::fabs(a[static_cast<size_t>(k) * static_cast<size_t>(n) + k]);
    for (int i = k + 1; i < n; ++i) {
      const double av = std::fabs(a[static_cast<size_t>(i) * static_cast<size_t>(n) + k]);
      if (av > max_abs) {
        max_abs = av;
        pivot = i;
      }
    }
    if (max_abs <= eps) return 0.0;
    if (pivot != k) {
      for (int j = k; j < n; ++j) {
        std::swap(a[static_cast<size_t>(k) * static_cast<size_t>(n) + j],
                  a[static_cast<size_t>(pivot) * static_cast<size_t>(n) + j]);
      }
      det_sign = -det_sign;
    }

    const double diag = a[static_cast<size_t>(k) * static_cast<size_t>(n) + k];
    det_val *= diag;
    if (std::fabs(diag) <= eps) return 0.0;
    for (int i = k + 1; i < n; ++i) {
      const double factor = a[static_cast<size_t>(i) * static_cast<size_t>(n) + k] / diag;
      if (factor == 0.0) continue;
      for (int j = k + 1; j < n; ++j) {
        a[static_cast<size_t>(i) * static_cast<size_t>(n) + j] -=
            factor * a[static_cast<size_t>(k) * static_cast<size_t>(n) + j];
      }
    }
  }
  return det_sign * det_val;
}

static double trace_matmul(const ColMat& a, const ColMat& b) {
  if (a.rows != a.cols || b.rows != b.cols || a.rows != b.rows) {
    throw std::runtime_error("trace_matmul requires square matrices with same size.");
  }
  const int d = a.rows;
  double tr = 0.0;
  for (int i = 0; i < d; ++i) {
    double s = 0.0;
    for (int k = 0; k < d; ++k) {
      s += a(i, k) * b(k, i);
    }
    tr += s;
  }
  return tr;
}

static void hugeglasso_sub(ColMat& S, ColMat& W, ColMat& T, int d, double ilambda, int& df, bool scr) {
  int i, j, k;
  int rss_idx, w_idx;

  int gap_int;
  double gap_ext, gap_act;
  const double thol_act = 1e-4;
  const double thol_ext = 1e-4;

  const int MAX_ITER_EXT = 100;
  const int MAX_ITER_INT = 10000;
  const int MAX_ITER_ACT = 10000;
  int iter_ext, iter_int, iter_act;

  std::vector<int> idx_a(static_cast<size_t>(d) * static_cast<size_t>(d), 0);
  std::vector<int> idx_i(static_cast<size_t>(d) * static_cast<size_t>(d), 0);
  std::vector<int> size_a(static_cast<size_t>(d), 0);
  std::vector<double> w1(static_cast<size_t>(d), 0.0);
  std::vector<double> ww(static_cast<size_t>(d), 0.0);

  int size_a_prev;
  int junk_a;

  double r;
  double tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
  const double neg_ilambda = -ilambda;

  for (i = 0; i < d; ++i) {
    int* idx_a_col = idx_a.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
    int* idx_i_col = idx_i.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
    double* T_col = T.col_ptr(i);
    const double* S_col = S.col_ptr(i);

    W(i, i) = S_col[i] + ilambda;
    size_a[static_cast<size_t>(i)] = 0;
    tmp1 = T_col[i];
    T_col[i] = 0;

    for (j = 0; j < d; ++j) {
      if (scr && std::fabs(S_col[j]) <= ilambda) {
        idx_i_col[j] = -1;
        T_col[j] = 0;
        continue;
      }
      if (T_col[j] != 0) {
        idx_a_col[size_a[static_cast<size_t>(i)]++] = j;
        idx_i_col[j] = -1;
        T_col[j] = -T_col[j] / tmp1;
      } else {
        idx_i_col[j] = 1;
      }
    }
    idx_i_col[i] = -1;
  }

  gap_ext = 1.0;
  iter_ext = 0;
  while (gap_ext > thol_ext && iter_ext < MAX_ITER_EXT) {
    tmp6 = 0.0;
    tmp5 = 0.0;
    for (i = 0; i < d; ++i) {
      int active_size = size_a[static_cast<size_t>(i)];
      int* idx_a_col = idx_a.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
      int* idx_i_col = idx_i.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
      double* T_col = T.col_ptr(i);
      const double* S_col = S.col_ptr(i);

      gap_int = 1;
      iter_int = 0;
      for (j = 0; j < d; ++j) ww[static_cast<size_t>(j)] = T_col[j];

      while (gap_int != 0 && iter_int < MAX_ITER_INT) {
        size_a_prev = active_size;
        for (j = 0; j < d; ++j) {
          if (idx_i_col[j] != -1) {
            r = S_col[j];
            for (k = 0; k < active_size; ++k) {
              rss_idx = idx_a_col[k];
              r -= W(rss_idx, j) * T_col[rss_idx];
            }

            double w_new = 0.0;
            if (r > ilambda) {
              w_new = (r - ilambda) / W(j, j);
              idx_a_col[active_size++] = j;
              idx_i_col[j] = -1;
            } else if (r < neg_ilambda) {
              w_new = (r + ilambda) / W(j, j);
              idx_a_col[active_size++] = j;
              idx_i_col[j] = -1;
            }

            w1[static_cast<size_t>(j)] = w_new;
            T_col[j] = w_new;
          }
        }
        gap_int = active_size - size_a_prev;

        gap_act = 1.0;
        iter_act = 0;
        while (gap_act > thol_act && iter_act < MAX_ITER_ACT) {
          tmp3 = 0.0;
          tmp4 = 0.0;
          for (j = 0; j < active_size; ++j) {
            w_idx = idx_a_col[j];
            r = S_col[w_idx] + T_col[w_idx] * W(w_idx, w_idx);
            for (k = 0; k < active_size; ++k) {
              rss_idx = idx_a_col[k];
              r -= W(rss_idx, w_idx) * T_col[rss_idx];
            }

            double w_new = 0.0;
            if (r > ilambda) {
              w_new = (r - ilambda) / W(w_idx, w_idx);
            } else if (r < neg_ilambda) {
              w_new = (r + ilambda) / W(w_idx, w_idx);
            }
            tmp4 += std::fabs(w_new);
            tmp3 += std::fabs(w_new - T_col[w_idx]);
            w1[static_cast<size_t>(w_idx)] = w_new;
            T_col[w_idx] = w_new;
          }
          gap_act = (tmp4 > 0.0) ? (tmp3 / tmp4) : 0.0;
          iter_act++;
        }

        junk_a = 0;
        for (j = 0; j < active_size; ++j) {
          w_idx = idx_a_col[j];
          if (w1[static_cast<size_t>(w_idx)] == 0.0) {
            junk_a++;
            idx_i_col[w_idx] = 1;
          } else {
            idx_a_col[j - junk_a] = w_idx;
          }
        }
        active_size -= junk_a;
        iter_int++;
      }
      size_a[static_cast<size_t>(i)] = active_size;

      std::vector<double> temp(static_cast<size_t>(d), 0.0);
      for (j = 0; j < d; ++j) {
        double s = 0.0;
        for (k = 0; k < d; ++k) s += W(j, k) * T_col[k];
        temp[static_cast<size_t>(j)] = s;
      }
      for (j = 0; j < i; ++j) {
        W(j, i) = temp[static_cast<size_t>(j)];
        W(i, j) = temp[static_cast<size_t>(j)];
      }
      for (j = i + 1; j < d; ++j) {
        W(j, i) = temp[static_cast<size_t>(j)];
        W(i, j) = temp[static_cast<size_t>(j)];
      }

      for (j = 0; j < d; ++j) tmp5 += std::fabs(ww[static_cast<size_t>(j)] - T_col[j]);
      tmp6 += tmp4;
    }
    gap_ext = (tmp6 > 0.0) ? (tmp5 / tmp6) : 0.0;
    iter_ext++;
  }

  for (i = 0; i < d; ++i) {
    double* T_col = T.col_ptr(i);
    tmp2 = 0.0;
    for (j = 0; j < d; ++j) tmp2 += W(j, i) * T_col[j];
    tmp2 -= W(i, i) * T_col[i];
    tmp1 = 1.0 / (W(i, i) - tmp2);
    for (j = 0; j < d; ++j) T_col[j] *= -tmp1;
    T_col[i] = tmp1;
  }
  for (i = 0; i < d; ++i) df += size_a[static_cast<size_t>(i)];
}

py::list threshold_path(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                        py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
  auto c = corr.unchecked<2>();
  auto l = lambdas.unchecked<1>();
  const py::ssize_t d = c.shape(0);
  if (d != c.shape(1)) throw std::runtime_error("corr must be square");

  py::list out;
  for (py::ssize_t k = 0; k < l.shape(0); ++k) {
    const double lam = l(k);
    const double thr = lam + 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(lam));
    auto mat = py::array_t<uint8_t>({d, d});
    auto m = mat.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < d; ++i) {
      for (py::ssize_t j = 0; j < d; ++j) {
        if (i == j) {
          m(i, j) = 0;
        } else {
          m(i, j) = (std::fabs(c(i, j)) > thr) ? 1 : 0;
        }
      }
    }
    out.append(mat);
  }
  return out;
}

py::array_t<double> sparsity_path(py::list matrices) {
  const py::ssize_t n = py::len(matrices);
  auto out = py::array_t<double>({n});
  auto o = out.mutable_unchecked<1>();
  for (py::ssize_t k = 0; k < n; ++k) {
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mat =
        py::reinterpret_borrow<py::array>(matrices[k]);
    auto m = mat.unchecked<2>();
    const py::ssize_t d = m.shape(0);
    if (d != m.shape(1)) throw std::runtime_error("each matrix must be square");
    double edges = 0.0;
    for (py::ssize_t i = 0; i < d; ++i) {
      for (py::ssize_t j = i + 1; j < d; ++j) {
        if (m(i, j) != 0) edges += 1.0;
      }
    }
    o(k) = (d <= 1) ? 0.0 : (2.0 * edges) / (static_cast<double>(d) * static_cast<double>(d - 1));
  }
  return out;
}

py::dict spmb_graph(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
  auto S = corr.unchecked<2>();
  auto lam = lambdas.unchecked<1>();
  const int d = static_cast<int>(S.shape(0));
  if (d <= 0 || d != static_cast<int>(S.shape(1))) throw std::runtime_error("corr must be square.");
  const int nlambda = static_cast<int>(lam.shape(0));
  if (nlambda <= 0) throw std::runtime_error("lambdas must be non-empty.");

  auto beta = py::array_t<double>({nlambda, d, d});
  auto df = py::array_t<double>({d, nlambda});
  std::fill_n(static_cast<double*>(beta.request().ptr), static_cast<size_t>(nlambda) * d * d, 0.0);
  std::fill_n(static_cast<double*>(df.request().ptr), static_cast<size_t>(d) * nlambda, 0.0);

  auto B = beta.mutable_unchecked<3>();
  auto DF = df.mutable_unchecked<2>();

  const double thol = 1e-4;
  const int MAX_ITER = 10000;
  std::vector<double> w0(static_cast<size_t>(d), 0.0);
  std::vector<double> w1(static_cast<size_t>(d), 0.0);
  std::vector<int> idx_a(static_cast<size_t>(d), 0);
  std::vector<int> idx_i(static_cast<size_t>(d), 0);

  for (int m = 0; m < d; ++m) {
    idx_i[static_cast<size_t>(m)] = 0;
    for (int j = 0; j < m; ++j) idx_i[static_cast<size_t>(j)] = 1;
    for (int j = m + 1; j < d; ++j) idx_i[static_cast<size_t>(j)] = 1;

    int size_a = 0;
    std::fill(w0.begin(), w0.end(), 0.0);
    std::fill(w1.begin(), w1.end(), 0.0);

    for (int i = 0; i < nlambda; ++i) {
      const double ilambda = lam(i);
      int gap_ext = 1;
      int iter_ext = 0;
      while (gap_ext != 0 && iter_ext < MAX_ITER) {
        int size_a_prev = size_a;
        for (int j = 0; j < d; ++j) {
          if (idx_i[static_cast<size_t>(j)] == 1) {
            double r = S(m, j);
            for (int k = 0; k < size_a; ++k) {
              const int rss_idx = idx_a[static_cast<size_t>(k)];
              r -= S(j, rss_idx) * w0[static_cast<size_t>(rss_idx)];
            }
            if (r > ilambda) {
              w1[static_cast<size_t>(j)] = r - ilambda;
              idx_a[static_cast<size_t>(size_a)] = j;
              size_a++;
              idx_i[static_cast<size_t>(j)] = 0;
            } else if (r < -ilambda) {
              w1[static_cast<size_t>(j)] = r + ilambda;
              idx_a[static_cast<size_t>(size_a)] = j;
              size_a++;
              idx_i[static_cast<size_t>(j)] = 0;
            } else {
              w1[static_cast<size_t>(j)] = 0.0;
            }
            w0[static_cast<size_t>(j)] = w1[static_cast<size_t>(j)];
          }
        }
        gap_ext = size_a - size_a_prev;

        double gap_int = 1.0;
        int iter_int = 0;
        while (gap_int > thol && iter_int < MAX_ITER) {
          double tmp1 = 0.0;
          double tmp2 = 0.0;
          for (int j = 0; j < size_a; ++j) {
            const int w_idx = idx_a[static_cast<size_t>(j)];
            double r = S(m, w_idx) + w0[static_cast<size_t>(w_idx)];
            for (int k = 0; k < size_a; ++k) {
              const int rss_idx = idx_a[static_cast<size_t>(k)];
              r -= S(w_idx, rss_idx) * w0[static_cast<size_t>(rss_idx)];
            }
            if (r > ilambda) {
              w1[static_cast<size_t>(w_idx)] = r - ilambda;
              tmp2 += std::fabs(w1[static_cast<size_t>(w_idx)]);
            } else if (r < -ilambda) {
              w1[static_cast<size_t>(w_idx)] = r + ilambda;
              tmp2 += std::fabs(w1[static_cast<size_t>(w_idx)]);
            } else {
              w1[static_cast<size_t>(w_idx)] = 0.0;
            }
            tmp1 += std::fabs(w1[static_cast<size_t>(w_idx)] - w0[static_cast<size_t>(w_idx)]);
            w0[static_cast<size_t>(w_idx)] = w1[static_cast<size_t>(w_idx)];
          }
          gap_int = (tmp2 > 0.0) ? (tmp1 / tmp2) : 0.0;
          iter_int++;
        }

        int junk_a = 0;
        for (int j = 0; j < size_a; ++j) {
          const int w_idx = idx_a[static_cast<size_t>(j)];
          if (w1[static_cast<size_t>(w_idx)] == 0.0) {
            junk_a++;
            idx_i[static_cast<size_t>(w_idx)] = 1;
          } else {
            idx_a[static_cast<size_t>(j - junk_a)] = w_idx;
          }
        }
        size_a -= junk_a;
        iter_ext++;
      }

      int nnz = 0;
      for (int j = 0; j < size_a; ++j) {
        const int w_idx = idx_a[static_cast<size_t>(j)];
        B(i, m, w_idx) = w1[static_cast<size_t>(w_idx)];
      }
      for (int j = 0; j < d; ++j) {
        if (std::fabs(B(i, m, j)) > 0.0) nnz++;
      }
      DF(m, i) = static_cast<double>(nnz);
    }
  }

  py::dict out;
  out["beta"] = beta;
  out["df"] = df;
  return out;
}

py::dict spmb_scr(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                  py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                  py::array_t<int, py::array::c_style | py::array::forcecast> idx_scr) {
  auto S = corr.unchecked<2>();
  auto lam = lambdas.unchecked<1>();
  auto idx = idx_scr.unchecked<2>();  // nscr x d
  const int d = static_cast<int>(S.shape(0));
  if (d <= 0 || d != static_cast<int>(S.shape(1))) throw std::runtime_error("corr must be square.");
  if (idx.shape(1) != d) throw std::runtime_error("idx_scr must have shape (nscr, d).");

  const int nscr = static_cast<int>(idx.shape(0));
  const int nlambda = static_cast<int>(lam.shape(0));
  if (nlambda <= 0) throw std::runtime_error("lambdas must be non-empty.");
  if (nscr <= 0) throw std::runtime_error("idx_scr must have positive rows.");

  auto beta = py::array_t<double>({nlambda, d, d});
  auto df = py::array_t<double>({d, nlambda});
  std::fill_n(static_cast<double*>(beta.request().ptr), static_cast<size_t>(nlambda) * d * d, 0.0);
  std::fill_n(static_cast<double*>(df.request().ptr), static_cast<size_t>(d) * nlambda, 0.0);
  auto B = beta.mutable_unchecked<3>();
  auto DF = df.mutable_unchecked<2>();

  const double thol = 1e-4;
  const int MAX_ITER = 10000;

  std::vector<double> w0(static_cast<size_t>(d), 0.0);
  std::vector<double> w1(static_cast<size_t>(d), 0.0);
  std::vector<int> idx_a(static_cast<size_t>(nscr), 0);
  std::vector<int> idx_i(static_cast<size_t>(nscr), 0);

  for (int m = 0; m < d; ++m) {
    int size_a = 0;
    for (int j = 0; j < nscr; ++j) idx_i[static_cast<size_t>(j)] = idx(j, m);
    std::fill(w0.begin(), w0.end(), 0.0);
    std::fill(w1.begin(), w1.end(), 0.0);

    for (int i = 0; i < nlambda; ++i) {
      const double ilambda = lam(i);
      int gap_ext = 1;
      int iter_ext = 0;
      while (iter_ext < MAX_ITER && gap_ext > 0) {
        int size_a_prev = size_a;
        for (int j = 0; j < nscr; ++j) {
          int w_idx = idx_i[static_cast<size_t>(j)];
          if (w_idx != -1) {
            double r = S(m, w_idx);
            for (int k = 0; k < size_a; ++k) {
              const int rss_idx = idx_a[static_cast<size_t>(k)];
              r -= S(w_idx, rss_idx) * w0[static_cast<size_t>(rss_idx)];
            }
            if (r > ilambda) {
              w1[static_cast<size_t>(w_idx)] = r - ilambda;
              idx_a[static_cast<size_t>(size_a)] = w_idx;
              size_a++;
              idx_i[static_cast<size_t>(j)] = -1;
            } else if (r < -ilambda) {
              w1[static_cast<size_t>(w_idx)] = r + ilambda;
              idx_a[static_cast<size_t>(size_a)] = w_idx;
              size_a++;
              idx_i[static_cast<size_t>(j)] = -1;
            } else {
              w1[static_cast<size_t>(w_idx)] = 0.0;
            }
            w0[static_cast<size_t>(w_idx)] = w1[static_cast<size_t>(w_idx)];
          }
        }

        gap_ext = size_a - size_a_prev;
        double gap_int = 1.0;
        int iter_int = 0;
        while (gap_int > thol && iter_int < MAX_ITER) {
          double tmp1 = 0.0;
          double tmp2 = 0.0;
          for (int j = 0; j < size_a; ++j) {
            const int w_idx = idx_a[static_cast<size_t>(j)];
            double r = S(m, w_idx) + w0[static_cast<size_t>(w_idx)];
            for (int k = 0; k < size_a; ++k) {
              const int rss_idx = idx_a[static_cast<size_t>(k)];
              r -= S(w_idx, rss_idx) * w0[static_cast<size_t>(rss_idx)];
            }
            if (r > ilambda) {
              w1[static_cast<size_t>(w_idx)] = r - ilambda;
              tmp2 += std::fabs(w1[static_cast<size_t>(w_idx)]);
            } else if (r < -ilambda) {
              w1[static_cast<size_t>(w_idx)] = r + ilambda;
              tmp2 += std::fabs(w1[static_cast<size_t>(w_idx)]);
            } else {
              w1[static_cast<size_t>(w_idx)] = 0.0;
            }
            tmp1 += std::fabs(w1[static_cast<size_t>(w_idx)] - w0[static_cast<size_t>(w_idx)]);
            w0[static_cast<size_t>(w_idx)] = w1[static_cast<size_t>(w_idx)];
          }
          gap_int = (tmp2 > 0.0) ? (tmp1 / tmp2) : 0.0;
          iter_int++;
        }
        iter_ext++;
      }

      for (int j = 0; j < size_a; ++j) {
        const int w_idx = idx_a[static_cast<size_t>(j)];
        B(i, m, w_idx) = w1[static_cast<size_t>(w_idx)];
      }
      int nnz = 0;
      for (int j = 0; j < d; ++j) {
        if (std::fabs(B(i, m, j)) > 0.0) nnz++;
      }
      DF(m, i) = static_cast<double>(nnz);
    }
  }

  py::dict out;
  out["beta"] = beta;
  out["df"] = df;
  return out;
}

py::dict spmb_graphsqrt(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                        py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
  auto X_in = data.unchecked<2>();
  auto lam = lambdas.unchecked<1>();
  const int n = static_cast<int>(X_in.shape(0));
  const int d = static_cast<int>(X_in.shape(1));
  const int nlambda = static_cast<int>(lam.shape(0));
  if (n <= 0 || d <= 0) throw std::runtime_error("data must be non-empty.");
  if (nlambda <= 0) throw std::runtime_error("lambdas must be non-empty.");

  std::vector<double> X(static_cast<size_t>(n) * static_cast<size_t>(d), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) X[static_cast<size_t>(i) * static_cast<size_t>(d) + j] = X_in(i, j);
  }
  auto x_at = [&](int r, int c) -> double { return X[static_cast<size_t>(r) * static_cast<size_t>(d) + c]; };

  auto beta = py::array_t<double>({nlambda, d, d});
  auto icov = py::array_t<double>({nlambda, d, d});
  auto df = py::array_t<double>({d, nlambda});
  std::fill_n(static_cast<double*>(beta.request().ptr), static_cast<size_t>(nlambda) * d * d, 0.0);
  std::fill_n(static_cast<double*>(icov.request().ptr), static_cast<size_t>(nlambda) * d * d, 0.0);
  std::fill_n(static_cast<double*>(df.request().ptr), static_cast<size_t>(d) * nlambda, 0.0);

  auto B = beta.mutable_unchecked<3>();
  auto I = icov.mutable_unchecked<3>();
  auto DF = df.mutable_unchecked<2>();

  const double prec = 1e-4;
  const int max_iter = 1000;
  const int num_relaxation_round = 3;
  const double eps = 1e-12;

  for (int m = 0; m < d; ++m) {
    std::vector<double> Xb(static_cast<size_t>(n), 0.0);
    std::vector<double> r(static_cast<size_t>(n), 0.0);
    std::vector<double> grad(static_cast<size_t>(d), 0.0);
    std::vector<double> gr(static_cast<size_t>(d), 0.0);
    std::vector<double> w1(static_cast<size_t>(d), 0.0);
    std::vector<double> Y(static_cast<size_t>(n), 0.0);
    for (int t = 0; t < n; ++t) Y[static_cast<size_t>(t)] = x_at(t, m);

    std::vector<double> Xb_master(static_cast<size_t>(n), 0.0);
    std::vector<double> w1_master(static_cast<size_t>(d), 0.0);
    std::vector<int> actset_indcat(static_cast<size_t>(d), 0);
    std::vector<int> actset_indcat_master(static_cast<size_t>(d), 0);
    std::vector<int> actset_idx;
    std::vector<double> old_coef(static_cast<size_t>(d), 0.0);
    std::vector<double> grad_master(static_cast<size_t>(d), 0.0);

    double L = 0.0;
    double sum_r2 = 0.0;
    double dev_thr = 0.0;
    auto refresh_residual = [&]() {
      sum_r2 = 0.0;
      for (int t = 0; t < n; ++t) {
        const double rt = r[static_cast<size_t>(t)];
        sum_r2 += rt * rt;
      }
      if (sum_r2 < eps) sum_r2 = eps;
      L = std::sqrt(sum_r2 / static_cast<double>(n));
      if (L < eps) L = eps;
      dev_thr = std::fabs(L) * prec;
    };

    for (int t = 0; t < n; ++t) r[static_cast<size_t>(t)] = Y[static_cast<size_t>(t)] - Xb[static_cast<size_t>(t)];
    refresh_residual();

    for (int j = 0; j < d; ++j) {
      double s = 0.0;
      for (int t = 0; t < n; ++t) s += r[static_cast<size_t>(t)] * x_at(t, j);
      grad[static_cast<size_t>(j)] = s / (static_cast<double>(n) * L);
      gr[static_cast<size_t>(j)] = std::fabs(grad[static_cast<size_t>(j)]);
      grad_master[static_cast<size_t>(j)] = gr[static_cast<size_t>(j)];
      w1_master[static_cast<size_t>(j)] = w1[static_cast<size_t>(j)];
      actset_indcat_master[static_cast<size_t>(j)] = 0;
    }
    for (int t = 0; t < n; ++t) Xb_master[static_cast<size_t>(t)] = Xb[static_cast<size_t>(t)];

    for (int i = 0; i < nlambda; ++i) {
      const double stage_lambda = lam(i);
      w1 = w1_master;
      Xb = Xb_master;
      for (int j = 0; j < d; ++j) {
        gr[static_cast<size_t>(j)] = grad_master[static_cast<size_t>(j)];
        actset_indcat[static_cast<size_t>(j)] = actset_indcat_master[static_cast<size_t>(j)];
      }

      const double threshold = (i > 0) ? (2.0 * lam(i) - lam(i - 1)) : (2.0 * lam(i));
      for (int j = 0; j < d; ++j) {
        if (j != m && gr[static_cast<size_t>(j)] > threshold) actset_indcat[static_cast<size_t>(j)] = 1;
      }
      for (int t = 0; t < n; ++t) r[static_cast<size_t>(t)] = Y[static_cast<size_t>(t)] - Xb[static_cast<size_t>(t)];
      refresh_residual();

      int loopcnt_level_0 = 0;
      while (loopcnt_level_0 < num_relaxation_round) {
        loopcnt_level_0++;

        int loopcnt_level_1 = 0;
        while (loopcnt_level_1 < max_iter) {
          loopcnt_level_1++;
          for (int j = 0; j < d; ++j) old_coef[static_cast<size_t>(j)] = w1[static_cast<size_t>(j)];
          refresh_residual();

          actset_idx.clear();
          auto update_coordinate = [&](int coord_idx) {
            double sum_wxx = 0.0;
            double sum_g = 0.0;
            for (int t = 0; t < n; ++t) {
              const double xv = x_at(t, coord_idx);
              const double rv = r[static_cast<size_t>(t)];
              const double wxx = (1.0 - (rv * rv) / sum_r2) * xv * xv;
              sum_wxx += wxx;
              sum_g += wxx * w1[static_cast<size_t>(coord_idx)] + rv * xv;
            }
            const double g = sum_g / (static_cast<double>(n) * L);
            const double a = sum_wxx / (static_cast<double>(n) * L);
            const double oldv = w1[static_cast<size_t>(coord_idx)];
            const double newv = (std::fabs(a) > eps) ? (threshold_l1(g, stage_lambda) / a) : 0.0;
            const double delta = newv - oldv;
            w1[static_cast<size_t>(coord_idx)] = newv;
            if (delta != 0.0) {
              for (int t = 0; t < n; ++t) {
                const double xv = x_at(t, coord_idx);
                Xb[static_cast<size_t>(t)] += delta * xv;
                r[static_cast<size_t>(t)] -= delta * xv;
              }
              refresh_residual();
            }
          };

          for (int j = 0; j < d; ++j) {
            if (j == m || !actset_indcat[static_cast<size_t>(j)]) continue;
            update_coordinate(j);
            if (std::fabs(w1[static_cast<size_t>(j)]) > 0.0) actset_idx.push_back(j);
          }

          int loopcnt_level_2 = 0;
          while (loopcnt_level_2 < max_iter) {
            loopcnt_level_2++;
            bool terminate_loop_level_2 = true;
            for (size_t k = 0; k < actset_idx.size(); ++k) {
              const int idx = actset_idx[k];
              const double old_w1 = w1[static_cast<size_t>(idx)];
              update_coordinate(idx);
              const double tmp_change = old_w1 - w1[static_cast<size_t>(idx)];
              double hsum = 0.0;
              for (int t = 0; t < n; ++t) {
                const double xv = x_at(t, idx);
                const double rv = r[static_cast<size_t>(t)];
                hsum += xv * xv * (1.0 - rv * rv / (L * L * static_cast<double>(n)));
              }
              const double h = std::fabs(hsum / (static_cast<double>(n) * L));
              const double local_change =
                  h * tmp_change * tmp_change / (2.0 * L * static_cast<double>(n));
              if (local_change > dev_thr) terminate_loop_level_2 = false;
            }
            if (terminate_loop_level_2) break;
          }

          bool terminate_loop_level_1 = true;
          for (size_t k = 0; k < actset_idx.size(); ++k) {
            const int idx = actset_idx[k];
            const double tmp_change = old_coef[static_cast<size_t>(idx)] - w1[static_cast<size_t>(idx)];
            double hsum = 0.0;
            for (int t = 0; t < n; ++t) {
              const double xv = x_at(t, idx);
              const double rv = r[static_cast<size_t>(t)];
              hsum += xv * xv * (1.0 - rv * rv / (L * L * static_cast<double>(n)));
            }
            const double h = std::fabs(hsum / (static_cast<double>(n) * L));
            const double local_change = h * tmp_change * tmp_change / (2.0 * L * static_cast<double>(n));
            if (local_change > dev_thr) terminate_loop_level_1 = false;
          }
          for (int t = 0; t < n; ++t) r[static_cast<size_t>(t)] = Y[static_cast<size_t>(t)] - Xb[static_cast<size_t>(t)];
          refresh_residual();
          if (terminate_loop_level_1) break;

          bool new_active_idx = false;
          for (int k = 0; k < d; ++k) {
            if (k == m || actset_indcat[static_cast<size_t>(k)] != 0) continue;
            double s = 0.0;
            for (int t = 0; t < n; ++t) s += r[static_cast<size_t>(t)] * x_at(t, k);
            grad[static_cast<size_t>(k)] = s / (static_cast<double>(n) * L);
            gr[static_cast<size_t>(k)] = std::fabs(grad[static_cast<size_t>(k)]);
            if (gr[static_cast<size_t>(k)] > stage_lambda) {
              actset_indcat[static_cast<size_t>(k)] = 1;
              new_active_idx = true;
            }
          }
          if (!new_active_idx) break;
        }

        if (loopcnt_level_0 == 1) {
          for (int j = 0; j < d; ++j) {
            w1_master[static_cast<size_t>(j)] = w1[static_cast<size_t>(j)];
            grad_master[static_cast<size_t>(j)] = gr[static_cast<size_t>(j)];
            actset_indcat_master[static_cast<size_t>(j)] = actset_indcat[static_cast<size_t>(j)];
          }
          for (int t = 0; t < n; ++t) Xb_master[static_cast<size_t>(t)] = Xb[static_cast<size_t>(t)];
        }
      }

      int nnz = 0;
      for (int j = 0; j < d; ++j) {
        if (j == m) continue;
        const double bv = w1[static_cast<size_t>(j)];
        if (std::fabs(bv) > 0.0) {
          B(i, m, j) = bv;
          nnz++;
        }
      }
      DF(m, i) = static_cast<double>(nnz);

      for (int t = 0; t < n; ++t) r[static_cast<size_t>(t)] = Y[static_cast<size_t>(t)] - Xb[static_cast<size_t>(t)];
      refresh_residual();
      const double tal = L;
      I(i, m, m) = (tal > 0.0) ? (1.0 / (tal * tal)) : 0.0;
      for (int j = 0; j < d; ++j) {
        if (j == m) continue;
        I(i, j, m) = -I(i, m, m) * w1[static_cast<size_t>(j)];
      }
    }
  }

  // Match huge::SPMBgraphsqrt behavior from R package build:
  // tmp_icov_p[i] = (tmp_icov_p[i].transpose() + tmp_icov_p[i]) / 2
  // That statement aliases RHS/LHS on Eigen matrices and results in a
  // column-major in-place update pattern (matrix may remain asymmetric).
  for (int i = 0; i < nlambda; ++i) {
    for (int c0 = 0; c0 < d; ++c0) {
      for (int r0 = 0; r0 < d; ++r0) {
        I(i, r0, c0) = 0.5 * (I(i, r0, c0) + I(i, c0, r0));
      }
    }
  }

  py::dict out;
  out["beta"] = beta;
  out["df"] = df;
  out["icov"] = icov;
  return out;
}

py::dict hugeglasso(py::array_t<double, py::array::c_style | py::array::forcecast> s,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lambdas, bool scr,
                    bool cov_output) {
  ColMat S = to_colmat(s, "S");
  if (S.rows != S.cols) throw std::runtime_error("S must be square.");
  auto lam = lambdas.unchecked<1>();
  const int d = S.rows;
  const int nlambda = static_cast<int>(lam.shape(0));

  auto loglik = py::array_t<double>({nlambda});
  auto sparsity = py::array_t<double>({nlambda});
  auto df = py::array_t<double>({nlambda});
  auto path = py::array_t<uint8_t>({nlambda, d, d});
  auto icov = py::array_t<double>({nlambda, d, d});

  std::fill_n(static_cast<double*>(loglik.request().ptr), static_cast<size_t>(std::max(nlambda, 0)), -static_cast<double>(d));
  std::fill_n(static_cast<double*>(sparsity.request().ptr), static_cast<size_t>(std::max(nlambda, 0)), 0.0);
  std::fill_n(static_cast<double*>(df.request().ptr), static_cast<size_t>(std::max(nlambda, 0)), 0.0);
  std::fill_n(static_cast<uint8_t*>(path.request().ptr), static_cast<size_t>(std::max(nlambda, 0)) * d * d, 0);
  std::fill_n(static_cast<double*>(icov.request().ptr), static_cast<size_t>(std::max(nlambda, 0)) * d * d, 0.0);

  py::array_t<double> cov;
  if (cov_output) {
    cov = py::array_t<double>({nlambda, d, d});
    std::fill_n(static_cast<double*>(cov.request().ptr), static_cast<size_t>(std::max(nlambda, 0)) * d * d, 0.0);
  }

  auto LL = loglik.mutable_unchecked<1>();
  auto SP = sparsity.mutable_unchecked<1>();
  auto DF = df.mutable_unchecked<1>();
  auto P = path.mutable_unchecked<3>();
  auto I = icov.mutable_unchecked<3>();

  if (nlambda > 0) {
    std::vector<double> s_diag(static_cast<size_t>(d), 0.0);
    for (int i = 0; i < d; ++i) s_diag[static_cast<size_t>(i)] = S(i, i);

    std::vector<ColMat> tmp_icov(static_cast<size_t>(nlambda), ColMat(d, d));
    std::vector<ColMat> tmp_path(static_cast<size_t>(nlambda), ColMat(d, d));
    std::vector<ColMat> tmp_cov;
    if (cov_output) tmp_cov.assign(static_cast<size_t>(nlambda), ColMat(d, d));

    const ColMat* prev_icov_ptr = nullptr;
    const ColMat* prev_cov_ptr = nullptr;
    ColMat prev_cov_buffer(d, d);

    bool zero_sol = true;
    const double sparsity_denom = (d > 1) ? (static_cast<double>(d) * static_cast<double>(d - 1)) : 1.0;

    for (int i = nlambda - 1; i >= 0; --i) {
      const double lambda_i = lam(i);
      std::vector<int> z;
      z.reserve(static_cast<size_t>(d));
      for (int row_i = 0; row_i < d; ++row_i) {
        int break_flag = 0;
        for (int col_i = 0; col_i < d; ++col_i) {
          if (break_flag > 1) break;
          if (S(row_i, col_i) > lambda_i || S(row_i, col_i) < -lambda_i) break_flag++;
        }
        if (break_flag > 1) z.push_back(row_i);
      }

      const int q = static_cast<int>(z.size());
      ColMat sub_S(q, q);
      ColMat sub_W(q, q);
      ColMat sub_T(q, q);
      int sub_df = 0;
      for (int ii = 0; ii < q; ++ii) {
        for (int jj = 0; jj < q; ++jj) {
          sub_S(ii, jj) = S(z[static_cast<size_t>(ii)], z[static_cast<size_t>(jj)]);
          if (zero_sol || prev_cov_ptr == nullptr || prev_icov_ptr == nullptr) {
            sub_W(ii, jj) = sub_S(ii, jj);
            sub_T(ii, jj) = (ii == jj) ? 1.0 : 0.0;
          } else {
            sub_W(ii, jj) = (*prev_cov_ptr)(z[static_cast<size_t>(ii)], z[static_cast<size_t>(jj)]);
            sub_T(ii, jj) = (*prev_icov_ptr)(z[static_cast<size_t>(ii)], z[static_cast<size_t>(jj)]);
          }
        }
      }

      if (q > 0) {
        hugeglasso_sub(sub_S, sub_W, sub_T, q, lambda_i, sub_df, scr);
        zero_sol = false;
      } else {
        zero_sol = true;
      }

      ColMat& cur_icov = tmp_icov[static_cast<size_t>(i)];
      ColMat& cur_path = tmp_path[static_cast<size_t>(i)];
      ColMat* cur_cov_ptr = nullptr;
      ColMat local_cov(d, d);
      if (cov_output) {
        cur_cov_ptr = &tmp_cov[static_cast<size_t>(i)];
      } else {
        cur_cov_ptr = &prev_cov_buffer;
      }

      cur_icov.set_zero();
      cur_path.set_zero();
      cur_cov_ptr->set_zero();
      for (int j = 0; j < d; ++j) {
        cur_icov(j, j) = 1.0 / (s_diag[static_cast<size_t>(j)] + lambda_i);
        (*cur_cov_ptr)(j, j) = s_diag[static_cast<size_t>(j)] + lambda_i;
      }

      if (!zero_sol) {
        for (int ii = 0; ii < q; ++ii) {
          for (int jj = 0; jj < q; ++jj) {
            const int ri = z[static_cast<size_t>(ii)];
            const int cj = z[static_cast<size_t>(jj)];
            cur_icov(ri, cj) = sub_T(ii, jj);
            (*cur_cov_ptr)(ri, cj) = sub_W(ii, jj);
            cur_path(ri, cj) = (ii == jj) ? 0.0 : ((sub_T(ii, jj) == 0.0) ? 0.0 : 1.0);
          }
        }
        SP(i) = static_cast<double>(sub_df) / sparsity_denom;
        DF(i) = static_cast<double>(sub_df) / 2.0;
        const double det_t = det_gaussian_elimination(sub_T);
        const double tr = trace_matmul(sub_T, sub_S);
        LL(i) = (det_t > 0.0) ? (std::log(det_t) - tr - static_cast<double>(d - q))
                              : (-std::numeric_limits<double>::infinity());
      }

      prev_icov_ptr = &cur_icov;
      prev_cov_ptr = cur_cov_ptr;
    }

    for (int k = 0; k < nlambda; ++k) {
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
          P(k, i, j) = static_cast<uint8_t>(tmp_path[static_cast<size_t>(k)](i, j) != 0.0);
          I(k, i, j) = tmp_icov[static_cast<size_t>(k)](i, j);
        }
      }
    }
    if (cov_output) {
      auto C = cov.mutable_unchecked<3>();
      for (int k = 0; k < nlambda; ++k) {
        for (int i = 0; i < d; ++i) {
          for (int j = 0; j < d; ++j) {
            C(k, i, j) = tmp_cov[static_cast<size_t>(k)](i, j);
          }
        }
      }
    }
  }

  py::dict out;
  out["path"] = path;
  out["icov"] = icov;
  out["loglik"] = loglik;
  out["sparsity"] = sparsity;
  out["df"] = df;
  if (cov_output) {
    out["cov"] = cov;
  } else {
    out["cov"] = py::none();
  }
  return out;
}

double ric(py::array_t<double, py::array::c_style | py::array::forcecast> x,
           py::array_t<int, py::array::c_style | py::array::forcecast> r) {
  auto X = x.unchecked<2>();
  auto R = r.unchecked<1>();
  const int n = static_cast<int>(X.shape(0));
  const int d = static_cast<int>(X.shape(1));
  const int t = static_cast<int>(R.shape(0));
  if (d <= 1 || n <= 0 || t <= 0) return 0.0;

  double lambda_min = std::numeric_limits<double>::infinity();
  for (int i = 0; i < t; ++i) {
    int tmp_r = R(i);
    if (tmp_r < 0) tmp_r = 0;
    if (tmp_r > n) tmp_r = n;
    const int split = n - tmp_r;
    double lambda_max = 0.0;
    for (int j = 0; j < d; ++j) {
      for (int k = j + 1; k < d; ++k) {
        double tmp = 0.0;
        for (int m = 0; m < split; ++m) {
          tmp += X(m + tmp_r, j) * X(m, k);
        }
        for (int m = split; m < n; ++m) {
          tmp += X(m - split, j) * X(m, k);
        }
        tmp = std::fabs(tmp);
        if (tmp > lambda_max) lambda_max = tmp;
      }
    }
    if (lambda_max < lambda_min) lambda_min = lambda_max;
  }
  if (!std::isfinite(lambda_min)) return 0.0;
  return lambda_min;
}

py::array_t<uint8_t> sfgen(int d0, int d, py::object seed_obj = py::none()) {
  if (d <= 0 || d0 <= 0 || d0 > d) throw std::runtime_error("Invalid input: require 0 < d0 <= d.");

  auto G = py::array_t<uint8_t>({d, d});
  auto g = G.mutable_unchecked<2>();
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) g(i, j) = 0;
  }

  std::vector<int> degree(static_cast<size_t>(d), 0);
  for (int i = 0; i < (d0 - 1); ++i) {
    g(i + 1, i) = 1;
    g(i, i + 1) = 1;
  }
  g(d0 - 1, 0) = 1;
  g(0, d0 - 1) = 1;
  for (int i = 0; i < d0; ++i) degree[static_cast<size_t>(i)] = 2;
  int total = 2 * d0;

  std::mt19937_64 rng;
  if (seed_obj.is_none()) {
    std::random_device rd;
    rng.seed(static_cast<uint64_t>(rd()));
  } else {
    rng.seed(static_cast<uint64_t>(seed_obj.cast<unsigned long long>()));
  }
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  for (int i = d0; i < d; ++i) {
    const double x = static_cast<double>(total) * unif(rng);
    int tmp = 0;
    int j = 0;
    while (tmp < x && j < i) {
      tmp += degree[static_cast<size_t>(j)];
      j++;
    }
    if (j > 0) j = j - 1;
    g(j, i) = 1;
    g(i, j) = 1;
    total += 2;
    degree[static_cast<size_t>(j)]++;
    degree[static_cast<size_t>(i)]++;
  }
  return G;
}

PYBIND11_MODULE(_native_core, m) {
  m.doc() = "pyhuge native C++ kernels (ported from huge src/*.cpp)";
  m.def("threshold_path", &threshold_path, py::arg("corr"), py::arg("lambdas"),
        "Build adjacency path by correlation thresholding.");
  m.def("sparsity_path", &sparsity_path, py::arg("matrices"), "Compute sparsity sequence from adjacency matrices.");
  m.def("spmb_graph", &spmb_graph, py::arg("corr"), py::arg("lambdas"),
        "MB graph path core (lossless screening).");
  m.def("spmb_scr", &spmb_scr, py::arg("corr"), py::arg("lambdas"), py::arg("idx_scr"),
        "MB graph path core (lossy screening).");
  m.def("spmb_graphsqrt", &spmb_graphsqrt, py::arg("data"), py::arg("lambdas"),
        "TIGER graph path core.");
  m.def("hugeglasso", &hugeglasso, py::arg("s"), py::arg("lambdas"), py::arg("scr") = false,
        py::arg("cov_output") = false, "Graphical lasso path core.");
  m.def("ric", &ric, py::arg("x"), py::arg("r"), "Rotation information criterion core.");
  m.def("sfgen", &sfgen, py::arg("d0"), py::arg("d"), py::arg("seed") = py::none(),
        "Scale-free graph generator core.");
}
