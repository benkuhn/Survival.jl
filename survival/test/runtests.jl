using Survival
using RDatasets
using ForwardDiff
using Base.Test

#### Test melanoma data

# We use a high epsilon because Harrell's RMS doesn't report results
# to high precision...
EPS = 1e-4
OPTIMAL_PARAMS = [0.4481, 0.0168, -0.1026, 0.1003, 1.1946]

melanoma = dataset("boot", "melanoma")
melanoma[:Surv] = DataArray(survec(float(melanoma[:Time]),
                                   melanoma[:Status] .== 1))
formula = Surv ~ 0 + Sex + Age + Year + Thickness + Ulcer

# obtain sorted X, y to test internal methods
mf = ModelFrame(formula, melanoma)
mm = ModelMatrix(mf)
X = mm.m
y = model_response(mf)
perm = sortperm(y, by=(surv)->surv.time)
X = X[perm,:]
y = y[perm]

# check that likelihood functions match Harrell's computations, and
# that handwritten gradient and Hessian line up with autodiff's
# results

# We test at the following two parameter settings
test_sets = [(zeros(Float64, 5), 283.1992),
             (OPTIMAL_PARAMS, 260.9995)]

lhfn = (params) -> Survival.coxlh(X, y, params)
autograd = forwarddiff_gradient(lhfn, Float64, fadtype=:typed)
autohess = forwarddiff_hessian(lhfn, Float64, fadtype=:typed)

for (params, target)=test_sets
    lh = Survival.coxlh(X, y, params)
    @test_approx_eq_eps target lh EPS

    grad = zeros(params)
    Survival.coxgrad!(X, y, params, grad)
    
    othergrad = autograd(params)
    for j in 1:length(params)
        @test_approx_eq_eps grad[j] othergrad[j] EPS
    end

    hess = zeros(Float64, length(params), length(params))
    Survival.coxhess!(X, y, params, hess)
    otherhess = autohess(params)
    for j in 1:length(otherhess)
        @test_approx_eq_eps hess[j] otherhess[j] EPS
    end
end

# Final test: fit a whole model and make sure it converges

model = fit(CoxPHModel, formula, melanoma)

@test Optim.converged(model.model.optresults)
@test_approx_eq_eps model.model.log_like 260.9995 EPS

for i=1:length(OPTIMAL_PARAMS)
    @test_approx_eq_eps model.model.params[i] OPTIMAL_PARAMS[i] EPS
end

### Test coeftable

coefs = coeftable(model)
# Output from R:
#
#           Coef    S.E.   Wald Z Pr(>|Z|)
# sex        0.4481 0.2669  1.68  0.0931
# age        0.0168 0.0086  1.96  0.0501
# year      -0.1026 0.0610 -1.68  0.0927
# thickness  0.1003 0.0382  2.63  0.0087
# ulcer      1.1946 0.3093  3.86  0.0001
target = [0.0931, 0.0501, 0.0927, 0.0087, 0.0001]
for i=1:length(target)
    @test_approx_eq_eps coefs.mat[i, coefs.pvalcol] target[i] 0.0001
end
