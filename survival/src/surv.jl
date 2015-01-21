type Surv{T}
    time::T
    event::Bool
end

function survec{T}(times::AbstractArray{T}, events)
    return [Surv{T}(time, event) for (time, event) = zip(times, events)]
end

time(s::Surv) = s.time
event(s::Surv) = s.event
