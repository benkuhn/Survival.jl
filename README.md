# Survival analysis in Julia

Writing basic survival analysis models in Julia to get a feel for the language.

```
julia> using Survival, RDatasets

julia> melanoma = dataset("boot", "melanoma");

julia> melanoma[:Surv] = DataArray(survec(float(melanoma[:Time]),
                                          melanoma[:Status] .== 1));

julia> model = fit(CoxPHModel, Surv ~ 0 + Sex + Age + Year + Thickness
                   + Ulcer, melanoma);

julia> model.model.params
5-element Array{Float64,1}:
  0.448121
  0.0168054
 -0.102566
  0.100312
  1.19455
```

The model still doesn't implement most of the StatisticalModel
interface (e.g. coefficient tables), so you probably shouldn't try to
use it for anything serious. But hey, it estimates the same
coefficients as R.
