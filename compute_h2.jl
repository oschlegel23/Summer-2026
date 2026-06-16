# Compute H2 in the Hamiltonian.
function compute_h2(xsamp, nmodes)
	h2 = 0
	for k in 1:nmodes
		h2 += k^2 * (xsamp[k]^2 + xsamp[k+nmodes]^2)
	end
	return 2*pi*h2
end