// Separate test translation unit: static helpers remain private in production.
// This executable must not also link a compiled huge_core library.
#include "huge_core.cpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
int checked = 0;
void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error(label);
}
huge::Matrix identity(int n) {
    huge::Matrix value(n, n);
    for (int i = 0; i < n; ++i) value(i, i) = 1.0;
    return value;
}
void compare(const huge::Matrix& w, const huge::Matrix& t,
             const std::string& label) {
    const auto original_w = w.v, original_t = t.v;
    const double expected = huge::inverse_residual_dense(w, t);
    const double actual = huge::inverse_residual_inf(w, t);
    require(std::isfinite(actual) == std::isfinite(expected), label + ": finite gate");
    for (double threshold : {0.005, 0.01})
        require((actual > threshold) == (expected > threshold), label + ": threshold gate");
    if (std::isfinite(expected))
        require(std::fabs(actual - expected) <= 1e-12 * std::max(1.0, expected),
                label + ": residual magnitude");
    // Raw memcmp is intentional: NaN fixtures are immutable too.
    require(std::memcmp(w.v.data(), original_w.data(), w.v.size()*sizeof(double)) == 0 &&
            std::memcmp(t.v.data(), original_t.data(), t.v.size()*sizeof(double)) == 0,
            label + ": mutated operand");
    ++checked;
}
void cancellation(int n, double value, const std::string& label) {
    auto w = identity(n), t = identity(n);
    // Exactly WT = I + value*E_02; its (0,2) dot is 1+value-1.
    // These are deliberately nonsymmetric matrices.
    w(0,1) = 1.0; w(0,2) = -1.0; w(1,2) = -value;
    t(0,1) = -1.0; t(0,2) = 1.0; t(1,2) = value;
    compare(w, t, label);
}
}
int main() {
    try {
        const int n = 512;
        for (int dimension : {511, 512, 513}) {
            const auto matrix = identity(dimension);
            compare(matrix, matrix, "identity dimension " + std::to_string(dimension));
        }
        for (double threshold : {0.005, 0.01}) {
            for (double value : {std::nextafter(threshold, 0.0), threshold,
                    std::nextafter(threshold, 1.0), threshold-1e-6, threshold+1e-6})
                cancellation(n, value, "cancellation near certification threshold");
            auto w = identity(n), t = identity(n);
            t(0,1) = threshold;
            compare(w, t, "exact representable off-diagonal threshold");
        }
        {
            auto w = identity(n), t = identity(n);
            t(0,1) = t(0,2) = t(0,3) = 0.002;
            compare(w, t, "row norm differs from column norm");
            require(std::fabs(huge::inverse_residual_inf(w,t)-0.006) < 1e-15,
                    "independent infinity norm oracle");
        }
        for (int count : {8192, 8193}) {
            auto w = identity(n), t = identity(n);
            int remaining = count - n;
            for (int col = 0; col < n; ++col)
                for (int row = 0; row < n && remaining > 0; ++row)
                    if (row != col) { t(row,col) = 1e-6; --remaining; }
            compare(w, t, "density edge " + std::to_string(count));
        }
        for (const auto& scales : {std::pair<double,double>{1e200,1e-200},
                {1e-200,1e200}, {1e200,1e200}, {1e-200,1e-200}, {1.0,0.0}}) {
            auto w = identity(n), t = identity(n);
            for (int i = 0; i < n; ++i) { w(i,i) = scales.first; t(i,i) = scales.second; }
            compare(w, t, "extreme scale");
        }
        {
            auto w = identity(n), t = identity(n);
            w(0,0) = 1e200; w(1,1) = 1e-200;
            t(0,0) = 1e-200; t(1,1) = 1e200;
            compare(w,t,"mixed scales with overflowing global bound");
        }
        {
            auto w = identity(n), t = identity(n);
            std::fill(w.v.begin(), w.v.end(), 1e305);
            for (int j = 0; j < n; ++j)
                for (int k = 1; k < 4; ++k) t((j+k)%n,j) = 1.0;
            compare(w,t,"finite products but overflowing absolute row sum");
        }
        for (double bad : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
            auto w = identity(n), t = identity(n);
            w(0,0) = bad;
            compare(w,t,"nonfinite covariance");
            w(0,0) = 1.0; t(0,0) = bad;
            compare(w,t,"nonfinite precision");
        }
        std::cout << checked << " residual boundary fixtures passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
