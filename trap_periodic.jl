function trap_periodic(fvals, x)
    h = x[2] - x[1]  
    return h * sum(fvals)
end