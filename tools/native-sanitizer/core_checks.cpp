#include <huge/huge_core.h>
#include <huge/blas_config.h>

#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double atol, double rtol,
           const std::string& message) {
    require(std::isfinite(actual) &&
                std::fabs(actual - expected) <= atol + rtol * std::fabs(expected),
            message);
}

template <typename Function>
void invalid(Function function, const std::string& message) {
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

huge::Matrix samples(int n, int d) {
    std::mt19937 random(91);
    std::normal_distribution<double> gaussian;
    huge::Matrix x(n, d);
    for (double& value : x.v) value = gaussian(random);
    return x;
}

huge::Matrix correlation(const huge::Matrix& x) {
    huge::Matrix z = x;
    for (int j = 0; j < z.cols; ++j) {
        double mean = 0.0, norm = 0.0;
        for (int i = 0; i < z.rows; ++i) mean += z(i, j);
        mean /= z.rows;
        for (int i = 0; i < z.rows; ++i) {
            z(i, j) -= mean;
            norm += z(i, j) * z(i, j);
        }
        for (int i = 0; i < z.rows; ++i) z(i, j) /= std::sqrt(norm);
    }
    huge::Matrix corr(z.cols, z.cols);
    for (int j = 0; j < z.cols; ++j) {
        corr(j, j) = 1.0;
        for (int i = 0; i < j; ++i) {
            double dot = 0.0;
            for (int k = 0; k < z.rows; ++k) dot += z(k, i) * z(k, j);
            corr(i, j) = corr(j, i) = dot;
        }
    }
    return corr;
}

void compare_columns(const std::vector<huge::ColResult>& left,
                     const std::vector<huge::ColResult>& right) {
    require(left.size() == right.size(), "column count differs across workers");
    for (size_t i = 0; i < left.size(); ++i)
        require(left[i].indices == right[i].indices && left[i].vals == right[i].vals,
                "coefficients differ across workers");
}

void mb_boundaries() {
    const double penalty = 0.005;
    const double lambda[] = {0.25, penalty, penalty};
    for (int d : {1, 16, 17, 40, 257, 258}) {
        huge::Matrix corr(d, d);
        corr(0, 0) = 1.0;
        std::vector<double> expected(d - 1);
        for (int j = 1; j < d; ++j) {
            expected[j - 1] = ((j - 1) % 2 ? -1.0 : 1.0) *
                (0.005 + 0.01 * ((j - 1) % 7) / 6.0);
            for (int k = 1; k < d; ++k)
                corr(j, k) = std::pow(0.2, std::abs(j - k));
        }
        for (int j = 1; j < d; ++j) {
            double value = penalty * (expected[j - 1] > 0.0 ? 1.0 : -1.0);
            for (int k = 1; k < d; ++k) value += corr(j, k) * expected[k - 1];
            corr(j, 0) = corr(0, j) = value;
        }
        for (bool screened : {false, true}) {
            if (screened && d == 1) continue;
            std::vector<int> indices(static_cast<size_t>(d) * (d - 1));
            for (int m = 0; m < d; ++m) {
                int offset = 0;
                for (int j = d - 1; j >= 0; --j)
                    if (j != m) indices[static_cast<size_t>(m) * (d - 1) + offset++] = j;
            }
            const auto fit = screened
                ? huge::mb_scr(corr.v.data(), d, lambda, 3, indices.data(), d - 1)
                : huge::mb(corr.v.data(), d, lambda, 3);
            require(!fit.hit_max_iter, "analytical MB problem failed convergence");
            std::vector<double> beta(static_cast<size_t>(3) * d, 0.0);
            for (int m = 0; m < d; ++m) {
                const auto& column = fit.columns[m];
                require(column.indices.size() == column.vals.size(), "MB index/value size");
                int previous = -1;
                for (size_t k = 0; k < column.indices.size(); ++k) {
                    int index = column.indices[k];
                    require(index > previous && index >= 0 && index < 3 * d &&
                                index % d != m && std::isfinite(column.vals[k]),
                            "MB sparse index ordering or bounds");
                    previous = index;
                    if (m == 0) beta[index] = column.vals[k];
                }
            }
            for (int j = 0; j < d; ++j) close(beta[j], 0.0, 0.0, 0.0, "MB empty first point");
            for (int path : {1, 2})
                for (int j = 1; j < d; ++j)
                    close(beta[path * d + j], expected[j - 1], 1e-6, 0.0,
                          "MB known signed solution");
        }
    }
}

void glasso_checks() {
    for (double scale : {1e-100, 1.0, 1e100}) {
        const double corr[] = {scale, 0.5 * scale, 0.5 * scale, scale};
        const double lambda[] = {0.7 * scale, 0.1 * scale, 0.1 * scale};
        for (bool screened : {false, true}) {
            const auto full = huge::glasso(corr, 2, lambda, 3, screened, true);
            const auto compact = huge::glasso_compact(corr, 2, lambda, 3, screened, false);
            require(compact.path.empty() && compact.cov.empty(), "glasso compact allocation contract");
            require(full.df == std::vector<int>({0, 1, 1}), "glasso analytical edge counts");
            require(full.loglik == compact.loglik, "glasso compact likelihood parity");
            for (int path = 0; path < 3; ++path) {
                require(full.icov[path].v == compact.icov[path].v, "glasso compact precision parity");
                double diagonal = path == 0 ? 1.7 : 1.1;
                double offdiagonal = path == 0 ? 0.0 : 0.4;
                double determinant = diagonal * diagonal - offdiagonal * offdiagonal;
                close(full.icov[path](0, 0) * scale, diagonal / determinant, 1e-12, 1e-10,
                      "glasso analytical diagonal");
                close(full.icov[path](0, 1) * scale, -offdiagonal / determinant, 1e-12, 1e-10,
                      "glasso analytical offdiagonal");
                require(full.icov[path](0, 1) == full.icov[path](1, 0), "glasso exact symmetry");
                require(std::isfinite(full.loglik[path]), "glasso finite scaled likelihood");
            }
        }
    }
}

void mb_kkt_checks() {
    const auto corr = correlation(samples(24, 48));
    const double lambda[] = {0.5, 0.2, 0.1};
    const auto fit = huge::mb(corr.v.data(), 48, lambda, 3);
    require(!fit.hit_max_iter, "MB rank-deficient test did not converge");
    for (int m = 0; m < 48; ++m) {
        huge::Matrix coefficients(48, 3);
        const auto& column = fit.columns[m];
        for (size_t k = 0; k < column.indices.size(); ++k)
            coefficients.v.at(column.indices[k]) = column.vals[k];
        for (int path = 0; path < 3; ++path)
            for (int j = 0; j < 48; ++j) {
                if (j == m) continue;
                double gradient = corr(j, m);
                for (int k = 0; k < 48; ++k)
                    gradient -= corr(j, k) * coefficients(k, path);
                double beta = coefficients(j, path);
                if (beta == 0.0)
                    require(std::fabs(gradient) <= lambda[path] + 1e-8, "MB inactive KKT");
                else
                    close(gradient, beta > 0.0 ? lambda[path] : -lambda[path],
                          3e-4, 0.0, "MB active KKT");
            }
    }
}

void glasso_markov_concurrency() {
    const int d = 512;
    const double rho = 0.7, penalty = 0.55;
    huge::Matrix covariance(d, d);
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i)
            covariance(i, j) = std::pow(rho, std::abs(i - j));
    const auto original = covariance.v;
    const auto reference = huge::glasso(covariance.v.data(), d, &penalty,
                                       1, false, true);
    const double a = 1.0 + penalty;
    const double r = (rho - penalty) / a;
    const double denominator = a * (1.0 - r * r);
    require(reference.df == std::vector<int>{d - 1}, "Markov chain support");
    for (int j = 0; j < d; ++j) {
        const double expected_diagonal = (j == 0 || j == d - 1)
            ? 1.0 / denominator : (1.0 + r * r) / denominator;
        close(reference.icov[0](j, j), expected_diagonal, 2e-6, 2e-4,
              "Markov chain precision diagonal");
        if (j > 0)
            close(reference.icov[0](j - 1, j), -r / denominator, 2e-6, 2e-4,
                  "Markov chain precision off-diagonal");
    }
    std::vector<std::future<void>> workers;
    for (int host = 0; host < 2; ++host)
        workers.push_back(std::async(std::launch::async, [&] {
            const auto actual = huge::glasso(covariance.v.data(), d, &penalty,
                                            1, false, true);
            require(actual.icov[0].v == reference.icov[0].v,
                    "sparse residual precision host-thread parity");
            require(actual.cov[0].v == reference.cov[0].v,
                    "sparse residual covariance host-thread parity");
            require(actual.loglik == reference.loglik && actual.df == reference.df,
                    "sparse residual metadata host-thread parity");
        }));
    for (auto& worker : workers) worker.get();
    require(covariance.v == original, "sparse residual mutated shared covariance");
}

void tiger_checks() {
    for (int d : {1, 7}) {
        huge::Matrix identity(d, d);
        for (int j = 0; j < d; ++j) identity(j, j) = 1.0;
        const double path[] = {0.8, 0.2, 0.2};
        const auto empty = huge::tiger_fit(identity.v.data(), d, d, true, path, 3, 0.1);
        require(empty.icov.size() == 3, "TIGER full empty precision path");
        for (const auto& precision : empty.icov)
            for (int j = 0; j < d; ++j)
                for (int i = 0; i < d; ++i)
                    require(i == j ? precision(i, j) == 1.0 :
                            (precision(i, j) == 0.0 && std::signbit(precision(i, j))),
                            "TIGER precision signed-zero contract");
    }
    const double corr[] = {1.0, 0.5, 0.5, 1.0};
    const double lambda[] = {0.6, 0.2, 0.2};
    const auto fit = huge::tiger_fit(corr, 2, 2, true, lambda, 3, 0.1);
    require(!fit.hit_max_iter && !fit.path_truncated, "TIGER analytic convergence");
    const double tau = std::sqrt(0.75 / 0.96);
    const double beta = 0.5 - 0.2 * tau;
    for (int path : {1, 2}) {
        close(fit.icov[path](0, 0), 1.0 / (tau * tau), 2e-6, 0.0, "TIGER analytical diagonal");
        close(fit.icov[path](1, 0), -beta / (tau * tau), 2e-6, 0.0, "TIGER analytical offdiagonal");
        require(fit.icov[path](0, 1) == fit.icov[path](1, 0), "TIGER exact symmetry");
    }
    const double singular[] = {1.0, 1.0, 1.0, 1.0};
    const auto truncated = huge::tiger_fit(singular, 2, 2, true, nullptr, 6, 0.1);
    require(truncated.path_truncated && truncated.hit_max_iter && !truncated.lambda.empty(),
            "TIGER singular path must report certified truncation");
    auto raw = samples(40, 4);
    auto extreme = raw;
    const double scales[] = {1e-200, 1e200, 1e-100, 1e100};
    for (int j = 0; j < raw.cols; ++j)
        for (int i = 0; i < raw.rows; ++i) extreme(i, j) *= scales[j];
    const double raw_lambda[] = {0.6, 0.3};
    const auto reference = huge::tiger_fit(raw.v.data(), 40, 4, false, raw_lambda, 2, 0.1);
    const auto scaled = huge::tiger_fit(extreme.v.data(), 40, 4, false, raw_lambda, 2, 0.1);
    for (int path = 0; path < 2; ++path)
        for (size_t i = 0; i < reference.icov[path].v.size(); ++i)
            close(scaled.icov[path].v[i], reference.icov[path].v[i], 1e-10, 1e-10,
                  "TIGER extreme raw scales");
}

void tiger_truncated_prefix_checks() {
    const auto corr = correlation(samples(24, 48));
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif
    const auto reference = huge::tiger_fit(corr.v.data(), 48, 48, true, nullptr, 20, 0.05);
    require(reference.hit_max_iter && reference.path_truncated &&
                reference.lambda.size() > 1 && reference.lambda.size() < 20,
            "TIGER random rank-deficient path should retain a nontrivial prefix");

    auto compare = [&](const huge::TigerResult& actual) {
        require(actual.lambda == reference.lambda &&
                    actual.hit_max_iter == reference.hit_max_iter &&
                    actual.path_truncated == reference.path_truncated,
                "TIGER common prefix depends on worker scheduling");
        compare_columns(actual.columns, reference.columns);
        require(actual.icov.size() == reference.icov.size(), "TIGER precision prefix length");
        for (size_t i = 0; i < reference.icov.size(); ++i)
            require(actual.icov[i].v.size() == reference.icov[i].v.size() &&
                    std::memcmp(actual.icov[i].v.data(), reference.icov[i].v.data(),
                                reference.icov[i].v.size() * sizeof(double)) == 0,
                    "TIGER certified precision bits depend on worker scheduling");
    };

    for (int workers : {1, 2, 4}) {
        #ifdef _OPENMP
        omp_set_num_threads(workers);
        #else
        (void)workers;
        #endif
        for (int repeat = 0; repeat < 3; ++repeat)
            compare(huge::tiger_fit(corr.v.data(), 48, 48, true, nullptr, 20, 0.05));
    }
    std::vector<std::future<void>> concurrent;
    for (int host = 0; host < 4; ++host)
        concurrent.push_back(std::async(std::launch::async, [&] {
            #ifdef _OPENMP
            omp_set_num_threads(2);
            #endif
            for (int repeat = 0; repeat < 3; ++repeat)
                compare(huge::tiger_fit(corr.v.data(), 48, 48, true, nullptr, 20, 0.05));
        }));
    for (auto& host : concurrent) host.get();

    const auto replay = huge::tiger_fit(corr.v.data(), 48, 48, true,
                                      reference.lambda.data(), reference.lambda.size(), 0.05);
    require(!replay.hit_max_iter && !replay.path_truncated,
            "TIGER explicit certified prefix should succeed without truncation");
    compare_columns(replay.columns, reference.columns);
    for (size_t i = 0; i < reference.icov.size(); ++i)
        require(replay.icov[i].v.size() == reference.icov[i].v.size() &&
                std::memcmp(replay.icov[i].v.data(), reference.icov[i].v.data(),
                            reference.icov[i].v.size() * sizeof(double)) == 0,
                "TIGER explicit prefix replay changed precision bits");

    auto failing_path = reference.lambda;
    failing_path.push_back(1e-6);
    bool rejected = false;
    try {
        huge::tiger_fit(corr.v.data(), 48, 48, true,
                        failing_path.data(), failing_path.size(), 0.05);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("supplied lambda") != std::string::npos;
    }
    require(rejected, "TIGER explicit uncertified suffix must still raise");
}

void invalid_inputs() {
    const double corr[] = {1.0, 0.2, 0.2, 1.0};
    const double lambda[] = {0.5};
    invalid([&] { huge::mb(nullptr, 2, lambda, 1); }, "MB null matrix");
    invalid([&] { huge::mb(corr, 0, lambda, 1); }, "MB zero dimension");
    invalid([&] { huge::glasso(corr, 2, nullptr, 1, false, false); }, "glasso null lambda");
    invalid([&] { huge::glasso(corr, 2, lambda, 0, false, false); }, "glasso empty lambda");
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        invalid([&] { huge::mb(corr, 2, &bad, 1); }, "MB unsafe lambda");
        invalid([&] { huge::glasso(corr, 2, &bad, 1, false, false); }, "glasso unsafe lambda");
        invalid([&] { huge::tiger_fit(corr, 2, 2, true, &bad, 1, 0.1); }, "TIGER unsafe lambda");
    }
    const double increasing[] = {0.1, 0.5};
    invalid([&] { huge::mb(corr, 2, increasing, 2); }, "MB increasing path");
    invalid([&] { huge::tiger_fit(corr, 2, 2, true, increasing, 2, 0.1); }, "TIGER increasing path");
    const int valid_screen[] = {1, 0};
    invalid([&] { huge::mb_scr(corr, 2, lambda, 1, nullptr, 1); }, "MB null screen");
    invalid([&] { huge::mb_scr(corr, 2, lambda, 1, valid_screen, 0); }, "MB empty screen");
    for (int bad : {-1, 0, 2}) {
        const int screen[] = {bad, 0};
        invalid([&] { huge::mb_scr(corr, 2, lambda, 1, screen, 1); }, "MB invalid screen index");
    }
    const double identity[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const int duplicates[] = {1, 1, 0, 2, 0, 1};
    invalid([&] { huge::mb_scr(identity, 3, lambda, 1, duplicates, 2); }, "MB duplicate screen indices");
    const double asym[] = {1.0, 0.1, 0.2, 1.0};
    const double indefinite[] = {1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0};
    invalid([&] { huge::tiger_fit(asym, 2, 2, true, lambda, 1, 0.1); }, "TIGER asymmetric covariance");
    invalid([&] { huge::tiger_fit(indefinite, 3, 3, true, lambda, 1, 0.1); }, "TIGER indefinite covariance");
    invalid([&] { huge::tiger_fit(nullptr, 2, 2, false, lambda, 1, 0.1); }, "TIGER null input");
    invalid([&] { huge::tiger_fit(corr, 1, 2, false, lambda, 1, 0.1); }, "TIGER too few observations");
    invalid([&] { huge::tiger_fit(corr, 1, 2, true, lambda, 1, 0.1); }, "TIGER nonsquare covariance");
    invalid([&] { huge::tiger_fit(corr, 2, 2, true, nullptr, 0, 0.1); }, "TIGER empty generated path");
    invalid([&] { huge::tiger_fit(corr, 2, 2, true, nullptr, 2, 0.0); }, "TIGER zero lambda ratio");
    const double constant[] = {1.0, 1.0, 1.0, 1.0};
    invalid([&] { huge::tiger_fit(constant, 2, 2, false, lambda, 1, 0.1); }, "TIGER constant data");
}

// Freeze the original full-matrix BLAS shape for the rounding-band contract.
// A scalar oracle cannot prescribe the same zero decision on every BLAS.
double ric_full_reference(const huge::Matrix& x, const std::vector<int>& rotations) {
    const int n = x.rows, d = x.cols, one = 1;
    if (n <= 0 || d <= 1 || rotations.empty()) return 0.0;
    const char trans = 'T', no_trans = 'N';
    const double zero = 0.0, unit = 1.0;
    const double eps = std::numeric_limits<double>::epsilon();
    const double gamma = n * eps / (1.0 - n * eps);
    std::vector<double> norm(d), product(static_cast<size_t>(d) * d);
    for (int j = 0; j < d; ++j) {
        const double* column = x.v.data() + static_cast<size_t>(j) * n;
        norm[j] = std::sqrt(ddot_(&n, column, &one, column, &one));
    }
    double result = std::numeric_limits<double>::infinity();
    for (int rotation : rotations) {
        rotation = std::max(0, std::min(n, rotation));
        const int split = n - rotation;
        dgemm_(&trans, &no_trans, &d, &d, &split, &unit,
               x.v.data() + rotation, &n, x.v.data(), &n, &zero, product.data(), &d);
        dgemm_(&trans, &no_trans, &d, &d, &rotation, &unit,
               x.v.data(), &n, x.v.data() + split, &n, &unit, product.data(), &d);
        double maximum = 0.0;
        for (int k = 1; k < d; ++k) for (int j = 0; j < k; ++j) {
            const double value = std::fabs(product[static_cast<size_t>(k) * d + j]);
            double bound = gamma * norm[j] * norm[k];
            bound += eps * bound + eps * value;
            if (value <= bound) continue;
            if (value > maximum) maximum = value;
        }
        result = std::min(result, maximum);
    }
    return std::isfinite(result) ? result : 0.0;
}

void ric_block_checks() {
    const std::vector<int> rotations = {0, 8, 17};
    for (int d : {1, 127, 128, 129, 130, 257, 258}) {
        const auto x = samples(17, d);
        const auto before = x.v;
        close(huge::ric(x.v.data(), 17, d, rotations.data(), rotations.size()),
              ric_full_reference(x, rotations), 0.0, 2e-13,
              "RIC full/partial panel and rotation endpoint reference");
        require(x.v == before, "RIC must not mutate its input");
    }

    // Eight meaningful Walsh columns expose a real zero-classification
    // regression from an unguarded panel on Accelerate. Use this BLAS's old
    // full shape as the oracle, including at other scales and rotations.
    huge::Matrix ambiguous(16, 258);
    for (int d : {130, 258}) for (double scale : {1e-200, 1e-100, 1.0, 1e100, 1e200}) {
        huge::Matrix x(16, d);
        const int columns[] = {0, 1, 2, 3, d-4, d-3, d-2, d-1};
        for (int row = 0; row < 16; ++row) {
            double basis[8];
            for (int column = 0; column < 8; ++column) {
                int value = row & (column + 1), parity = 0;
                for (; value; value >>= 1) parity ^= value & 1;
                basis[column] = parity ? -1.0 : 1.0;
                x(row, columns[column]) = scale * basis[column];
            }
            x(row, columns[7]) = scale * (basis[7] +
                16 * std::numeric_limits<double>::epsilon() * basis[6]);
        }
        if (d == 258 && scale == 1e100) ambiguous = x;
        for (int rotation : {0, 8, 16}) {
            const std::vector<int> selected = {rotation};
            close(huge::ric(x.v.data(), 16, d, &rotation, 1),
                  ric_full_reference(x, selected), 0.0, 0.0,
                  "RIC ambiguity and extreme-scale decisions match full BLAS");
        }
    }

    const auto x = samples(17, 258);
    const double expected = huge::ric(x.v.data(), 17, 258, rotations.data(), rotations.size());
    const std::vector<int> ambiguous_rotations = {0, 8, 16};
    const double ambiguous_expected = ric_full_reference(ambiguous, ambiguous_rotations);
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.push_back(std::async(std::launch::async, [&] {
            #ifdef _OPENMP
            omp_set_num_threads(2);
            #endif
            for (int repeat = 0; repeat < 3; ++repeat) {
                close(huge::ric(x.v.data(), 17, 258, rotations.data(), rotations.size()),
                      expected, 0.0, 0.0, "RIC panels have private worker scratch");
                close(huge::ric(ambiguous.v.data(), 16, 258,
                               ambiguous_rotations.data(), ambiguous_rotations.size()),
                      ambiguous_expected, 0.0, 0.0, "RIC fallback has private worker scratch");
            }
        }));
    for (auto& worker : workers) worker.get();
}

void ric_and_generator_checks() {
    const double x[] = {1, 2, 3, 4, 5, -2, 1, 0, 3, -1, 0, 1, 2, 1, -1};
    const int rotations[] = {0, 1, 3, 5};
    double expected = std::numeric_limits<double>::infinity();
    for (int rotation : rotations) {
        double maximum = 0.0;
        for (int j = 0; j < 3; ++j)
            for (int k = j + 1; k < 3; ++k) {
                double dot = 0.0;
                for (int row = 0; row < 5; ++row)
                    dot += x[j * 5 + (row + rotation) % 5] * x[k * 5 + row];
                maximum = std::max(maximum, std::fabs(dot));
            }
        expected = std::min(expected, maximum);
    }
    close(huge::ric(x, 5, 3, rotations, 4), expected, 1e-12, 0.0,
          "RIC agrees with scalar rotation reference");
    std::vector<double> random(35);
    for (size_t i = 0; i < random.size(); ++i)
        random[i] = i % 2 ? std::nextafter(1.0, 0.0) : 0.0;
    std::vector<int> graph(37 * 37);
    huge::sfgen(2, 37, graph.data(), random.data());
    int entries = 0;
    for (int j = 0; j < 37; ++j) {
        int degree = 0;
        for (int k = 0; k < 37; ++k) {
            int value = graph[j * 37 + k];
            require(value == 0 || value == 1, "scale-free binary entries");
            require(value == graph[k * 37 + j], "scale-free symmetry");
            degree += value;
        }
        require(graph[j * 37 + j] == 0 && degree > 0, "scale-free graph node invariant");
        entries += degree;
    }
    require(entries == 2 * 36, "scale-free graph edge count");
}

void concurrency_checks() {
    auto x = samples(32, 40);
    auto corr = correlation(x);
    const double lambda[] = {0.5, 0.2, 0.05};
    const auto reference = huge::mb(corr.v.data(), 40, lambda, 3);
    std::vector<int> screen(40 * 39);
    for (int m = 0; m < 40; ++m) {
        int position = 0;
        for (int j = 39; j >= 0; --j)
            if (j != m) screen[m * 39 + position++] = j;
    }
    const auto screen_reference = huge::mb_scr(corr.v.data(), 40, lambda, 3, screen.data(), 39);
    const int rotations[] = {0, 1, 7, 32};
    const double ric_reference = huge::ric(x.v.data(), 32, 40, rotations, 4);
    const double small[] = {1.0, 0.5, 0.5, 1.0};
    const auto glasso_reference = huge::glasso(small, 2, lambda, 3, false, true);
    const auto tiger_reference = huge::tiger_fit(small, 2, 2, true, lambda, 3, 0.1);
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.push_back(std::async(std::launch::async, [&] {
            #ifdef _OPENMP
            omp_set_num_threads(2);
            #endif
            for (int repeat = 0; repeat < 4; ++repeat) {
                const auto actual = huge::mb(corr.v.data(), 40, lambda, 3);
                compare_columns(actual.columns, reference.columns);
                compare_columns(huge::mb_scr(corr.v.data(), 40, lambda, 3, screen.data(), 39).columns,
                                screen_reference.columns);
                require(actual.hit_max_iter == reference.hit_max_iter, "MB threaded convergence flag");
                close(huge::ric(x.v.data(), 32, 40, rotations, 4), ric_reference, 0.0, 0.0,
                      "RIC concurrent shared-input parity");
                const auto glasso = huge::glasso(small, 2, lambda, 3, false, true);
                const auto tiger = huge::tiger_fit(small, 2, 2, true, lambda, 3, 0.1);
                compare_columns(tiger.columns, tiger_reference.columns);
                for (int path = 0; path < 3; ++path) {
                    require(glasso.icov[path].v == glasso_reference.icov[path].v,
                            "glasso concurrent shared-input parity");
                    require(tiger.icov[path].v == tiger_reference.icov[path].v,
                            "TIGER concurrent shared-input parity");
                }
            }
        }));
    for (auto& worker : workers) worker.get();
}

} // namespace

int main() {
    try {
        #if defined(HUGE_EXPECT_OPENMP) && !defined(_OPENMP)
        throw std::runtime_error("OpenMP was required but the harness compiled without it");
        #endif
        #ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(4);
        int team_size = 0;
        #pragma omp parallel reduction(max:team_size)
        { team_size = omp_get_num_threads(); }
        require(team_size >= 2, "OpenMP did not create multiple workers");
        std::cout << "OpenMP workers: " << team_size << '\n';
        #else
        std::cout << "Serial core; testing concurrent host threads\n";
        #endif
        mb_boundaries(); std::cout << "MB active boundaries and known solutions passed\n";
        mb_kkt_checks(); std::cout << "MB rank-deficient KKT checks passed\n";
        glasso_checks(); std::cout << "glasso analytical/scale/compact checks passed\n";
        glasso_markov_concurrency(); std::cout << "glasso Markov oracle and sparse residual concurrency passed\n";
        tiger_checks(); std::cout << "TIGER analytical/scale/singular checks passed\n";
        tiger_truncated_prefix_checks(); std::cout << "TIGER common-prefix scheduling/replay checks passed\n";
        invalid_inputs(); std::cout << "Invalid native inputs rejected\n";
        ric_block_checks(); std::cout << "RIC panels, ambiguity fallback, and shared-input checks passed\n";
        ric_and_generator_checks(); std::cout << "RIC scalar reference and generator checks passed\n";
        concurrency_checks(); std::cout << "Concurrent shared-input determinism passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Native check failed: " << error.what() << '\n';
        return 1;
    }
}
