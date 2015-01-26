module Survival

using StatsBase
using Optim
using DataArrays
using Distributions

export survec, fit, CoxResp, CoxPHModel, Surv

include("surv.jl")
include("coxph.jl")

end
