// pybind11 thin wrapper — delegates to huge::* core functions
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "huge/huge_core.h"

#include <cstdint>
#include <random>
#include <vector>

namespace py = pybind11;

// Helper: copy row-major numpy 2D array to column-major vector
static std::vector<double> numpy_to_colmajor(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                                              int& rows, int& cols) {
    auto a = arr.unchecked<2>();
    rows = static_cast<int>(a.shape(0));
    cols = static_cast<int>(a.shape(1));
    std::vector<double> out(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[static_cast<size_t>(j) * rows + i] = a(i, j);
    return out;
}

// Helper: copy column-major huge::Matrix to numpy
static py::array_t<double> matrix_to_numpy(const huge::Matrix& m) {
    auto out = py::array_t<double>({m.rows, m.cols});
    auto o = out.mutable_unchecked<2>();
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            o(i, j) = m(i, j);
    return out;
}

// ---- Glasso binding ----

static py::dict py_hugeglasso(py::array_t<double, py::array::c_style | py::array::forcecast> s,
                               py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                               bool scr, bool cov_output) {
    int d, d2;
    auto S = numpy_to_colmajor(s, d, d2);
    if (d != d2) throw std::runtime_error("S must be square.");
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    auto res = huge::glasso(S.data(), d, lam_vec.data(), nlambda, scr, cov_output);

    // Convert results
    auto loglik = py::array_t<double>(nlambda);
    auto sparsity = py::array_t<double>(nlambda);
    auto df = py::array_t<double>(nlambda);
    auto path = py::array_t<uint8_t>({nlambda, d, d});
    auto icov_out = py::array_t<double>({nlambda, d, d});

    auto LL = loglik.mutable_unchecked<1>();
    auto SP = sparsity.mutable_unchecked<1>();
    auto DF = df.mutable_unchecked<1>();
    auto P = path.mutable_unchecked<3>();
    auto I = icov_out.mutable_unchecked<3>();

    for (int k = 0; k < nlambda; k++) {
        LL(k) = res.loglik[k];
        SP(k) = res.sparsity[k];
        DF(k) = static_cast<double>(res.df[k]);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                P(k, i, j) = (res.path[k](i, j) != 0.0) ? 1 : 0;
                I(k, i, j) = res.icov[k](i, j);
            }
    }

    py::dict out;
    out["path"] = path;
    out["icov"] = icov_out;
    out["loglik"] = loglik;
    out["sparsity"] = sparsity;
    out["df"] = df;
    if (cov_output) {
        auto cov = py::array_t<double>({nlambda, d, d});
        auto C = cov.mutable_unchecked<3>();
        for (int k = 0; k < nlambda; k++)
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    C(k, i, j) = res.cov[k](i, j);
        out["cov"] = cov;
    } else {
        out["cov"] = py::none();
    }
    return out;
}

// Helper: convert ColResult columns to dense beta[nlambda,d,d] + df[d,nlambda]
static std::pair<py::array_t<double>, py::array_t<double>>
columns_to_dense(const std::vector<huge::ColResult>& columns, int d, int nlambda) {
    auto beta = py::array_t<double>({nlambda, d, d});
    auto df = py::array_t<double>({d, nlambda});
    std::fill_n(static_cast<double*>(beta.request().ptr), static_cast<size_t>(nlambda) * d * d, 0.0);
    std::fill_n(static_cast<double*>(df.request().ptr), static_cast<size_t>(d) * nlambda, 0.0);
    auto B = beta.mutable_unchecked<3>();
    auto DF = df.mutable_unchecked<2>();

    for (int m = 0; m < d; m++) {
        const auto& col = columns[m];
        for (size_t j = 0; j < col.vals.size(); j++) {
            int encoded = col.indices[j];
            B(encoded / d, m, encoded % d) = col.vals[j];
        }
        for (int i = 0; i < nlambda; i++) {
            int nnz = 0;
            for (int j = 0; j < d; j++)
                if (std::fabs(B(i, m, j)) > 0.0) nnz++;
            DF(m, i) = static_cast<double>(nnz);
        }
    }
    return {beta, df};
}

// ---- MB binding ----

static py::dict py_spmb_graph(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                               py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
    int d, d2;
    auto S = numpy_to_colmajor(corr, d, d2);
    if (d != d2) throw std::runtime_error("corr must be square.");
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    auto res = huge::mb(S.data(), d, lam_vec.data(), nlambda);
    auto [beta, df] = columns_to_dense(res.columns, d, nlambda);

    py::dict out;
    out["beta"] = beta;
    out["df"] = df;
    return out;
}

// ---- MB with screening binding ----

static py::dict py_spmb_scr(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                              py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                              py::array_t<int, py::array::c_style | py::array::forcecast> idx_scr_arr) {
    int d, d2;
    auto S = numpy_to_colmajor(corr, d, d2);
    if (d != d2) throw std::runtime_error("corr must be square.");
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    auto idx = idx_scr_arr.unchecked<2>();
    int nscr = static_cast<int>(idx.shape(0));
    std::vector<int> idx_scr_cm(static_cast<size_t>(nscr) * d);
    for (int m = 0; m < d; m++)
        for (int j = 0; j < nscr; j++)
            idx_scr_cm[static_cast<size_t>(m) * nscr + j] = idx(j, m);

    auto res = huge::mb_scr(S.data(), d, lam_vec.data(), nlambda, idx_scr_cm.data(), nscr);
    auto [beta, df] = columns_to_dense(res.columns, d, nlambda);

    py::dict out;
    out["beta"] = beta;
    out["df"] = df;
    return out;
}

// ---- TIGER binding ----

static py::dict py_spmb_graphsqrt(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
    int n, d;
    auto X = numpy_to_colmajor(data, n, d);
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    auto res = huge::tiger(X.data(), n, d, lam_vec.data(), nlambda);
    auto [beta, df] = columns_to_dense(res.columns, d, nlambda);

    auto icov = py::array_t<double>({nlambda, d, d});
    auto I = icov.mutable_unchecked<3>();
    for (int k = 0; k < nlambda; k++)
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                I(k, i, j) = res.icov[k](i, j);

    py::dict out;
    out["beta"] = beta;
    out["df"] = df;
    out["icov"] = icov;
    return out;
}

// ---- Threshold path (pure Python-side, no core needed) ----

static py::list py_threshold_path(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
    auto c = corr.unchecked<2>();
    auto l = lambdas.unchecked<1>();
    const py::ssize_t d = c.shape(0);
    py::list out;
    for (py::ssize_t k = 0; k < l.shape(0); k++) {
        double lam = l(k);
        double thr = lam + 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(lam));
        auto mat = py::array_t<uint8_t>({d, d});
        auto m = mat.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < d; i++)
            for (py::ssize_t j = 0; j < d; j++)
                m(i, j) = (i != j && std::fabs(c(i, j)) > thr) ? 1 : 0;
        out.append(mat);
    }
    return out;
}

static py::array_t<double> py_sparsity_path(py::list matrices) {
    const py::ssize_t n = py::len(matrices);
    auto out = py::array_t<double>({n});
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t k = 0; k < n; k++) {
        auto mat = py::reinterpret_borrow<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(matrices[k]);
        auto m = mat.unchecked<2>();
        const py::ssize_t d = m.shape(0);
        double edges = 0;
        for (py::ssize_t i = 0; i < d; i++)
            for (py::ssize_t j = i + 1; j < d; j++)
                if (m(i, j) != 0) edges += 1.0;
        o(k) = (d <= 1) ? 0.0 : 2.0 * edges / (static_cast<double>(d) * (d - 1));
    }
    return out;
}

// ---- RIC binding ----

static double py_ric(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                      py::array_t<int, py::array::c_style | py::array::forcecast> r) {
    auto X = x.unchecked<2>();
    auto R = r.unchecked<1>();
    int n_rows = static_cast<int>(X.shape(0));
    int d = static_cast<int>(X.shape(1));
    int t = static_cast<int>(R.shape(0));

    // Convert to column-major
    std::vector<double> X_cm(static_cast<size_t>(n_rows) * d);
    for (int i = 0; i < n_rows; i++)
        for (int j = 0; j < d; j++)
            X_cm[static_cast<size_t>(j) * n_rows + i] = X(i, j);

    std::vector<int> r_vec(t);
    for (int i = 0; i < t; i++) r_vec[i] = R(i);

    double result = huge::ric(X_cm.data(), n_rows, d, r_vec.data(), t);
    if (!std::isfinite(result)) return 0.0;
    return result;
}

// ---- SFGen binding ----

static py::array_t<uint8_t> py_sfgen(int d0, int d, py::object seed_obj) {
    if (d <= 0 || d0 <= 0 || d0 > d) throw std::runtime_error("Invalid input: require 0 < d0 <= d.");

    int nrand = d - d0;
    std::vector<double> rands(nrand > 0 ? nrand : 0);

    std::mt19937_64 rng;
    if (seed_obj.is_none()) {
        std::random_device rd;
        rng.seed(static_cast<uint64_t>(rd()));
    } else {
        rng.seed(static_cast<uint64_t>(seed_obj.cast<unsigned long long>()));
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int i = 0; i < nrand; i++) rands[i] = unif(rng);

    std::vector<int> G(static_cast<size_t>(d) * d, 0);
    huge::sfgen(d0, d, G.data(), rands.data());

    // Convert to uint8 numpy
    auto out = py::array_t<uint8_t>({d, d});
    auto o = out.mutable_unchecked<2>();
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            o(i, j) = static_cast<uint8_t>(G[static_cast<size_t>(j) * d + i]);
    return out;
}

// ---- Module definition ----

PYBIND11_MODULE(_native_core, m) {
    m.doc() = "pyhuge native C++ kernels (shared core with R package)";
    m.def("threshold_path", &py_threshold_path, py::arg("corr"), py::arg("lambdas"),
          "Build adjacency path by correlation thresholding.");
    m.def("sparsity_path", &py_sparsity_path, py::arg("matrices"),
          "Compute sparsity sequence from adjacency matrices.");
    m.def("spmb_graph", &py_spmb_graph, py::arg("corr"), py::arg("lambdas"),
          "MB graph path core (lossless screening).");
    m.def("spmb_scr", &py_spmb_scr, py::arg("corr"), py::arg("lambdas"), py::arg("idx_scr"),
          "MB graph path core (lossy screening).");
    m.def("spmb_graphsqrt", &py_spmb_graphsqrt, py::arg("data"), py::arg("lambdas"),
          "TIGER graph path core.");
    m.def("hugeglasso", &py_hugeglasso, py::arg("s"), py::arg("lambdas"), py::arg("scr") = false,
          py::arg("cov_output") = false, "Graphical lasso path core.");
    m.def("ric", &py_ric, py::arg("x"), py::arg("r"),
          "Rotation information criterion core.");
    m.def("sfgen", &py_sfgen, py::arg("d0"), py::arg("d"), py::arg("seed") = py::none(),
          "Scale-free graph generator core.");
}
