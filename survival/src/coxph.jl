type CoxPHModel{T} <: RegressionModel
    y::Array{Surv{T}}
    null_like::T
    log_like::T
    invhess::Matrix{T}  # inverted hessian at the optimized params
    params::Vector{T}
    optresults::Optim.MultivariateOptimizationResults
end

function coxlh{T}(X::Matrix{T}, y::Array{Surv{T}}, params::Array)
    n = size(X, 1)
    jmin = 1 # minimum index with exit time equal to current time
    theta = exp(X * params)
    sum_theta = sum(theta) # sum_{j >= jmin} exp(x_j * params)
    log_like = 0
    for i=1:n
        if i > 1 && y[i].time > y[i-1].time
            # change sum_theta
            sum_theta -= sum(theta[jmin:i-1])
            jmin = i
        end
        if !y[i].event
            continue
        end
        log_like += dot(X[i,:], params)
        log_like -= log(sum_theta)
    end
    return -log_like
end

function addto(storage::Array, summand::Array)
    @assert size(storage) == size(summand) "$size(storage) != $size(summand)"
    for i=1:length(storage)
        storage[i] += summand[i]
    end
end

function coxgrad!{T}(X::Matrix{T}, y::Array{Surv{T}}, params::Array{T}, storage::Array{T})
    n = size(X, 1)
    jmin = 1 # minimum index with exit time equal to current time
    theta = exp(X * params)
    sum_theta = sum(theta)            # sum_{j >= jmin} exp(x_j * params)
    sum_theta_x = vec(sum(theta .* X, 1))  # sum_{j >= jmin} theta_j * X_j
    fill!(storage, 0)
    for i=1:n
        if i > 1 && y[i].time > y[i-1].time
            # change sum_theta
            span = jmin:i-1
            sum_theta -= sum(theta[span])
            sum_theta_x -= vec(sum(theta[span] .* X[span,:], 1))
            jmin = i
        end
        if !y[i].event
            continue
        end
        addto(storage, sum_theta_x / sum_theta - vec(X[i,:]))
    end
end

function coxhess!{T}(X::Matrix{T}, y::Array{Surv{T}}, params::Array{T}, storage::Matrix{T})
    # TODO this is symmetric...
    n = size(X, 1)
    jmin = 1 # minimum index with exit time equal to current time
    theta = exp(X * params)
    norm_x = sum(X .^ 2, 2)
    sum_theta = sum(theta)            # sum_{j >= jmin} exp(x_j * params)
    sum_theta_x = vec(sum(theta .* X, 1))  # sum_{j >= jmin} theta_j * X_j
    sum_theta_outer_x = transpose(theta .* X) * X # sum_{j >= jmin} theta_j * X_j
    fill!(storage, 0)
    for i=1:n
        if i > 1 && y[i].time > y[i-1].time
            # change sum_theta
            span = jmin:i-1
            sum_theta -= sum(theta[span])
            sum_theta_x -= vec(sum(theta[span] .* X[span,:], 1))
            sum_theta_outer_x -= transpose(theta[span] .* X[span,:]) * X[span,:]
            jmin = i
        end
        if !y[i].event
            continue
        end
        addto(storage,
              - (sum_theta_x * transpose(sum_theta_x))::Matrix{T} / (sum_theta ^ 2)
              +sum_theta_outer_x / sum_theta)
    end
end

function StatsBase.fit{T<:FloatingPoint}(::Type{CoxPHModel},
                                         X::Array{T, 2},
                                         y::Array{Surv{T}, 1};
                                         maxIter=1000)
    size(X, 1) == size(y, 1) || DimensionMismatch("number of rows in X and y must match")
    nparam = size(X, 2)
    perm = sortperm(y, by=time)
    y = y[perm]
    X = X[perm,:]

    #lh(params::Array{T}) = coxlh(X, y, params)
    #g!(params::Array{T}, storage::Array{T}) = coxgrad!(X, y, params, storage)
    #h!(params::Array{T}, storage::Array{T}) = coxhess!(X, y, params, storage)
    null_like = coxlh(X, y, zeros(T, nparam))
    lh(params) = coxlh(X, y, params)
    g!(params, storage) = coxgrad!(X, y, params, storage)
    h!(params, storage) = coxhess!(X, y, params, storage)
    #h! = forwarddiff_hessian!(lh, Float64, fadtype=:typed)
    d2 = DifferentiableFunction(lh, g!)
    d3 = TwiceDifferentiableFunction(lh, g!, h!)

    results = optimize(d3, zeros(T, nparam),
                       method=:newton,
                       #show_trace=true,
                       iterations=maxIter)
    Optim.converged(results) || warn("no convergence after $maxIter iterations")
    params = results.minimum
    log_like = lh(params)

    invhess = zeros(T, nparam, nparam)::Matrix{T}
    coxhess!(X, y, params, invhess)
    invhess = inv(invhess)
    
    CoxPHModel(y, null_like, log_like, invhess, params, results)
end

function StatsBase.coeftable{T}(m::CoxPHModel{T})
    # coef, se, z, P>|Z|, l95, u95
    COEF, SE, Z, P_GT_Z, L95, U95 = (1:6)
    colnms = ["Coef", "S.E.", "Z", "P>|Z|", "Lower 95%", "Upper 95%"]
    coef = m.params
    mat = zeros(T, length(coef), length(colnms))
    mat[:, COEF] = coef
    for i=1:length(coef)
        se = sqrt(m.invhess[i, i])
        mat[i, SE] = se
        mat[i, Z] = coef[i] / se
        mat[i, P_GT_Z] = cdf(Normal(), -abs(coef[i] / se)) * 2
        mat[i, L95] = coef[i] - 1.96 * se
        mat[i, U95] = coef[i] + 1.96 * se
    end

    rownms = [1:length(coef)]
    return CoefTable(mat, colnms, rownms, P_GT_Z)
end
