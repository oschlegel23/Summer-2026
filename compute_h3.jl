# Compute H3 in the Hamiltonian.
function compute_h3(xsamp, nmodes)
	h3 = 0
	for n = 2:nmodes
		uhn_conj = xsamp[n] + im*xsamp[n+nmodes]
		for k = 1:n-1
			h3 += real( uhn_conj * (xsamp[k]-im*xsamp[k+nmodes]) * (xsamp[n-k]-im*xsamp[n-k+nmodes]) )
		end
	end
	return 2*pi*h3
end
