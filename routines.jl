# Subroutines for KdV and Hamiltonian computations.


#---------------------------------------------------#
# NOTE: For these routines, it is assumed that uhat 
# does NOT contain the zero-mode, but we need to settle that.

# Compute the Hamiltonian component H2.
function ham2(uhat)
	h2 = 0.
	for k in 1:length(uhat)
		h2 += k^2 * abs2(uhat[k])
	end
	return 2*pi*h2
end

# Compute the Hamiltonian component H3.
function ham3(uhat)
	h3 = 0.
	for n = 2:length(uhat)
		for k = 1:n-1
			h3 += real( conj(uhat[n])*uhat[k]*uhat[n-k] )
		end
	end
	return 2*pi*h3
end

# Energy = 2*pi*sum_{k=1:kmax} uhat_k^2
compute_energy(uhat) = 2*pi*sum(abs2.(uhat))

#---------------------------------------------------#

# Compute the product of two funtions in Fourier space: w = u*v.
# NOTE: Here u, v, and w contain a zero-mode.
function dealias_product_direct(uhat, vhat)
	kmax = length(uhat) - 1
	what = zeros(ComplexF64, kmax+1)
	# Compute the coefficient what_k of the product w=u*v.
	for k in 0:kmax
		# Sum over (u,v) indices that are (+,+)
		sum1 = zero(ComplexF64)
		for j in 0:k 
			sum1 += uhat[j+1] * vhat[k-j+1]
		end
		# Sum over (u,v) indices that are (+,-)
		sum2 = zero(ComplexF64)
		for j in k+1:kmax
			sum2 += uhat[j+1] * conj(vhat[j-k+1])
		end
		# Sum over (u,v) indices that are (-,+)
		sum3 = zero(ComplexF64)
		for j in 1:kmax-k 
			sum3 += conj(uhat[j+1]) * vhat[k+j+1]
		end
		# Combine the three sums.
		what[k+1] = sum1 + sum2 + sum3
	end
	return what
end 

