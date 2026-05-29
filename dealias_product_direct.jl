function dealias_product_direct(u,v)

    K = length(u) - 1
    w = zeros(ComplexF64, K+1)

    for k in 0:K

        sum1 = zero(ComplexF64)
        for j in 0:k 
            sum1 += u[j+1] * v[k-j+1]
        end

        sum2 = zero(ComplexF64)
        for j in k+1:K
            sum2 += u[j+1] * conj(v[j-k+1])
        end

        sum3 = zero(ComplexF64)
        for j in 1:K-k 
            sum3 += conj(u[j+1]) * v[k+j+1]
        end

        w[k+1] = sum1 + sum2 + sum3

    end

    return w

end 




# hand check
#u = ComplexF64[1, 2 + 1im, 3]
# v = ComplexF64[4, 1 - 1im, 2 + 2im]

# expected = ComplexF64[18, 18 + 8im, 17 + 1im]

# w = dealias_product_direct(u, v)
# println("max error test 1 = ", maximum(abs.(w .- expected)))






# indep check
#K = 8                  # use modes 0..K
#N = 4 * (2K + 1)       
#x = 2π * (0:N-1) / N   # grid points on [0, 2π)

# u_modes = randn(ComplexF64, K + 1)
# v_modes = randn(ComplexF64, K + 1)
# u_modes[1] = real(u_modes[1])   
# v_modes[1] = real(v_modes[1])


# function to_phys(modes, N)
#     K = length(modes) - 1
#     full = zeros(ComplexF64, N)
#     full[1:K+1] = modes                              # positive modes 0..K
#     full[N-K+1:N] = conj.(reverse(modes[2:end]))     # negative modes -K..-1
#     return real(ifft(full) * N)
# end

# ux = to_phys(u_modes, N)
# vx = to_phys(v_modes, N)

# # multiply in physical space - u(x)·v(x)
# wx = ux .* vx

# # FFT to get its modes, keep only modes 0..K
# w_ref = (fft(wx) / N)[1:K+1]

# # call our function for comparison
# w_ours = dealias_product_direct(u_modes, v_modes)

# println("max error test 2 = ", maximum(abs.(w_ours .- w_ref)))