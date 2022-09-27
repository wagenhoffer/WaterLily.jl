### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ ed603e7a-1819-44b1-ab7c-4f3155046691
using Plots,WaterLily,StaticArrays, ForwardDiff

# ╔═╡ 6a533d64-db81-11ec-19ae-992ce0add6e0
# First mode dimensionless wavenumber. β² is the dimensionless ωₙ
β₁=0.596864π

# ╔═╡ a97c2f48-a56c-46bf-a5f1-5707acef7d81
# Mode shape
ψ(x,β=β₁) = cosh(β*x)-cos(β*x)+(cosh(β)+cos(β))*(sin(β*x)-sinh(β*x))/(sin(β)+sinh(β))

# ╔═╡ 3fdbbd59-6f10-4c62-bdb9-54e79de7f313
begin
	x = 0:0.01:1
	plot(x,ψ.(x))
end

# ╔═╡ 6cb27a55-f0b9-4f69-b1fa-feee73b60097
begin 
	struct O2state{T}
		p::Vector{T}
		v::Vector{T}
		t::Vector{T}
		function O2state(init::Vector{T}) where T
			new{T}([init[1]],[init[2]],[init[3]])
		end
	end
	import Base.push!
	push!(a::O2state,v::Vector) = (push!(a.p,v[1]);push!(a.v,v[2]);push!(a.t,v[3]);a)
	(a::O2state{T})(t) where T = a.p[end]+a.v[end]*(t-a.t[end])
	
	function step!(s::O2state,t;f=t->0,ωₙ=1,ζ=0)
		p₀,v₀,t₀ = s.p[end],s.v[end],s.t[end]
		a = f(t)-ωₙ^2*p₀-2ζ*v₀
		v = v₀+(t-t₀)*a
		p = p₀+(t-t₀)*v
		push!(s,[p,v,t])
	end
end

# ╔═╡ 80ce526f-4fd6-4f72-a5b2-4e682c23a8fd
begin
	A = O2state([0.01,0.07,0.])
	function foil(L=2^6; thk=2+√2, U=1, ν=1e-4)
		# fraction along length
		s(x) = clamp(x/L,0,1)
		
		# thickened line
		sdf(x,t) = √sum(abs2,x-L*SVector(s(x[1]),0.))-0.5thk

		# displacement
		map(x,t) = SVector(x[1]-L,x[2]-L-L*A(t*U/L)*ψ(s(x[1]-L)))
		
		# make the fish simulation
		return Simulation((3L+2,2L+2),zeros(2),L;U,ν,body=AutoBody(sdf,map))
	end

	function sim_step!(sim::Simulation,t_end)
		Fψ(t) = ∮pfds(sim.flow.p,sim.body,t*sim.L/sim.U,
					(x,t)->metrics(x,t,sim)[4])/sim.L^2
		t = WaterLily.time(sim)
	    while t < t_end*sim.L/sim.U
			ωₙ=1+0.3cos(2t*sim.U/sim.L)
			f(t) = ωₙ^2*π/128*cos(t)-Fψ(t)
			step!(A,t*sim.U/sim.L;f,ωₙ,ζ=0.25)
	        measure!(sim,t)
	        mom_step!(sim.flow,sim.pois) # evolve Flow
	        t += sim.flow.Δt[end]
	    end
	end
end

# ╔═╡ b2af83e0-fa2f-410f-8cef-31585d5d7999
begin
	function ∮pfds(p, body, t = 0, 
		f = (x,t) -> ForwardDiff.gradient(y -> body.sdf(y,t), x))
		R = inside(p)
		s = 0 .* f(WaterLily.loc(0,first(R)),t)
	    for I ∈ R
	        x = WaterLily.loc(0,I)
	        d = body.sdf(x,t)::Float64
	        abs(d) ≤ 1 && (s += f(x,t).*p[I]*WaterLily.kern(d))
	    end
	    return s
	end
	function metrics(x,t,sim)
		nx,ny = ForwardDiff.gradient(y -> sim.body.sdf(y,t), x)
		l = x ./ sim.L .- 1
		return SVector(nx,ny,l[1]*ny-nx*l[2],ψ(clamp(l[1],0,1))*ny)
	end
	function get_force(sim,t)
		sim_step!(sim,t)
		return ∮pfds(sim.flow.p,sim.body,t*sim.L/sim.U,
						(x,t)->metrics(x,t,sim))./(0.5*sim.L*sim.U^2)
	end
end

# ╔═╡ 50f00b57-2429-4a01-8b82-7c24bf331606
begin
	# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
	function plot_vorticity(sim)
		@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
		contourf(sim.flow.σ', 
			color=palette(:BuGn), clims=(-10,10),linewidth=0,
			aspect_ratio=:equal,legend=false,border=:none)
        contourf!(sim.flow.σ', 
			color=palette(:BuGn), clims=(-1,1),linewidth=0,
			aspect_ratio=:equal,legend=false,border=:none)
	end

	# make a gif over a swimming cycle
	swimmer = foil()
	N = 3; m = N*24
	cycle = range(0, (m-1)/(m)*N*2π, m)
	@gif for t ∈ sim_time(swimmer).+cycle
		sim_step!(swimmer,t)
		plot_vorticity(swimmer)
	end
end

# ╔═╡ 784f520d-a396-4bdb-8723-59a65fec29bd
begin	
	sim = deepcopy(swimmer)
	forces = [get_force(sim,t) for t ∈ sim_time(sim).+cycle]
	"got forces"
end

# ╔═╡ a0d9504b-8bab-4c5d-a879-ea0525149edd
scatter(range(0,(m-1)/m*N,m),reduce(hcat,forces)', 
	labels=["thrust" "side" "moment" "bending"], 
	xlabel="scaled time", ylabel="scaled action")

