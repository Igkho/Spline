#pragma once
#include "Spline.h"
#include <vector>
#include <stdexcept>

namespace spline {

// Interface class for complex function
template <class T>
class IComplexFunction {
public:
    IComplexFunction(const std::vector<const IBasicFunction<T> *> &funcs) : funcs_(funcs) {}

    // The pure virtual method for returning the value of a function
    virtual std::vector<T> operator()(const std::vector<T> &args) const = 0;

    // The pure virtual method for returning the derivative of a function
    virtual std::vector<std::vector<T>> Derivative(const std::vector<T> &args) const = 0;
protected:
    std::vector<const IBasicFunction<T> *> funcs_;
};

// The class for difference of two functions
template <class T>
class Difference: public IComplexFunction<T> {
public:
    Difference(const std::vector<const IBasicFunction<T> *> &funcs) : IComplexFunction<T>(funcs) {
        if (this->funcs_.size() != 2) {
            throw std::runtime_error("Difference should have two sub functions");
        }
    }

    // The method for difference calculation
    virtual std::vector<T> operator()(const std::vector<T> &args) const override {
        if (args.size() != 2) {
            throw std::runtime_error("Difference input should have two arguments");
        }
        auto f0 = (*this->funcs_[0])(args[0]);
        auto f1 = (*this->funcs_[1])(args[1]);
        return {f0[0] - f1[0], f0[1] - f1[1]};
    }

    // The method for difference derivative calculation
    virtual std::vector<std::vector<T>> Derivative(const std::vector<T> &args) const override {
        if (args.size() != 2) {
            throw std::runtime_error("Difference derivative input should have two arguments");
        }
        auto d0 = (*this->funcs_[0]).Derivative(args[0]);
        auto d1 = (*this->funcs_[1]).Derivative(args[1]);
        for (auto &d : d1) {
            d *= (T)-1;
        }
        return {d0, d1};
    }
};

// The class for L2 norm
template <class T>
class L2Norm: public IComplexFunction<T> {
public:
    L2Norm(const std::vector<const IBasicFunction<T> *> &funcs) : IComplexFunction<T>(funcs) {
        if (this->funcs_.size() != 2) {
            throw std::runtime_error("L2Norm should have two sub functions");
        }
    }

    // The method for L2Norm calculation
    virtual std::vector<T> operator()(const std::vector<T> &args) const override {
        if (args.size() != 2) {
            throw std::runtime_error("L2Norm input should have two arguments");
        }
        auto f0 = (*this->funcs_[0])(args[0]);
        auto f1 = (*this->funcs_[1])(args[1]);
        T norm = (T)0;
        for (size_t i = 0; i < f0.size(); ++i) {
            T delta = f0[i] - f1[i];
            norm += delta * delta;
        }
        return {norm};
    }

    // The method for L2Norm derivative calculation
    virtual std::vector<std::vector<T>> Derivative(const std::vector<T> &args) const override {
        if (args.size() != 2) {
            throw std::runtime_error("L2Norm derivative input should have two arguments");
        }
        auto d0 = this->funcs_[0]->Derivative(args[0]);
        auto d1 = this->funcs_[1]->Derivative(args[1]);
        auto f0 = (*this->funcs_[0])(args[0]);
        auto f1 = (*this->funcs_[1])(args[1]);
        T v0 = (T)0, v1 = (T)0;
        for (size_t i = 0; i < f0.size(); ++i) {
            T delta = f0[i] - f1[i];
            v0 += 2 * delta * d0[i];
            v1 -= 2 * delta * d1[i];
        }
        return {{v0, v1}};
    }
};

// Interface class for optimizers
template <class T>
class IOptimizer {
public:
    IOptimizer(const std::vector<const IBasicFunction<T> *> &funcs) : funcs_(funcs) {}

    // The pure virtual method for iterative optimization
    virtual std::vector<T> Optimize(const std::vector<T> &args,
                                    size_t max_iters) = 0;

protected:
    std::vector<const IBasicFunction<T> *> funcs_;

    // The pure virtual method for one optimization step
    virtual bool OptimizeStep(std::vector<T> &arg,
                              std::vector<std::vector<T>> &value) = 0;
};

// Newton-Raphson optimizer class
template <class T>
class NROptimizer: public IOptimizer<T> {
public:
    NROptimizer(const std::vector<const IBasicFunction<T> *> &funcs, T epsilon) :
        IOptimizer<T>(funcs), diff_(funcs), epsilon_(epsilon) {}

    virtual std::vector<T> Optimize(const std::vector<T> &args,
                                    size_t max_iters) override;

protected:
    const Difference<T> diff_;
    const T epsilon_;

    virtual bool OptimizeStep(std::vector<T> &args,
                              std::vector<std::vector<T>> &values) override;
};

// RMSProp optimizer class
template <class T>
class RMSPropOptimizer: public IOptimizer<T> {
public:
    RMSPropOptimizer(const std::vector<const IBasicFunction<T> *> &funcs,
                     T alpha, T beta, T epsilon) :
        IOptimizer<T>(funcs), l2n_(funcs), alpha_(alpha), beta_(beta), epsilon_(epsilon) {}

    virtual std::vector<T> Optimize(const std::vector<T> &args,
                                    size_t max_iters) override;

protected:
    std::vector<T> v_;
    const L2Norm<T> l2n_;
    T alpha_, beta_, epsilon_;

    virtual bool OptimizeStep(std::vector<T> &args,
                              std::vector<std::vector<T>> &values) override;
};

// Interface class for curves search initializer
template <class T>
class SearchInitializer {
public:
    static constexpr T MAX_DPARAM = (T)0.5;
    static constexpr T MIN_DPARAM = (T)0.01;

    SearchInitializer(const std::vector<const BasicCurve<T> *> &curves) : curves_(curves) {}
    std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<T>>>
        GetSearchPointsAndInitResults(T search_ratio = 1) const;

protected:
    const std::vector<const BasicCurve<T> *> curves_;
    std::vector<std::vector<T>> GetSearchGrid(const T ratio = 1) const;

};

} // namespace spline
