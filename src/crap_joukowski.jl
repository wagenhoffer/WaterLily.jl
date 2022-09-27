# crapp attempt at plotting joukowski transforms

using LinearAlgebra
using Plots
using WaterLily

a = 1 

R = 1.1 *a



n_pos = 110
β = 0   #chamber
θ = range(-β,2π-β, n_pos)
den = @. (a + R*cos(θ))^2 + R^2*sin(θ)^2
x = @. a + R*(cos(θ)-1) + a/den + R*cos(θ)/den
y = @. R*sin(θ) - R*sin(θ)/den

plot(x,y)

z(β,R,a) = a .* ( 1 .+ R/a.*(exp.(1im*θ(β)) .- exp(-1im*β)))

ζ = z(β,R,a) .+ 1 ./ z(θ,β,R,a)
T = maximum(imag(ζ))*2.0/(maximum(real(ζ))-minimum(real(ζ)))
plot(real(ζ),imag(ζ), aspect_ratio=:equal)

xs = (real(ζ) .+ 2.0) 
ys = imag(ζ) 
T = maximum(ys)*2.0/(maximum(xs)-minimum(xs))
plot(xs,ys, aspect_ratio=:equal)
# plot!(x/4.0,y/4.0, aspect_ratio=:equal)
L = 1 #length
# fraction along length
s(x) = clamp(x/L,0,1)

# thickened line
sdf(x,t) = √sum(abs2,x-L*SVector(s(x[1]),0.))-0.5thk

function segment_sdf(x,y) 
    s = clamp(x,0,1)         # distance along the segment
    y = y-shift              # shift laterally
    sdf = √sum(abs2,(x-s,y)) # line segment SDF
    return sdf     # subtract thickness
end
grid = -1:0.5:3
contourf(grid,grid,segment_sdf,clim=(-1,2),linewidth=0)
contour!(grid,grid,segment_sdf,levels=[0],color=:black) 

begin
	function segment_sdf(x,y) 
		s = clamp(x,0,1)         # distance along the segment
		y = y-shift              # shift laterally
		sdf = √sum(abs2,(x-s,y)) # line segment SDF
		return sdf-T*y     # subtract thickness
	end
	grid = -1:0.05:2
	contourf(grid,grid,segment_sdf,clim=(-1,2),linewidth=0)
	contour!(grid,grid,segment_sdf,levels=[0],color=:black) # zero contour
end

using WaterLily
using StaticArrays

function block(L=2^5;Re=250,U=1,amp=0,ϵ=0.5,thk=2ϵ+√2)
    # Set viscosity
    ν=U*L/Re

    # Create dynamic block geometry
    function sdf(x,t)
        y = x .- SVector(0.,clamp(x[2],-L/2,L/2))
        √sum(abs2,y)-thk/2
    end
    function map(x,t)
        α = amp*cos(t*U/L); R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        R * (x.-SVector(3L+L*sin(t*U/L)+0.01,4L))
    end
    body = AutoBody(sdf,map)

    Simulation((6L+2,6L+2),zeros(2),L;U,ν,body,ϵ)
end

using BenchmarkTools
test() = @benchmark sim_step!(sim,π/4,remeasure=true) setup=(sim=block())

include("TwoD_plots.jl")
# sim_gif!(block();duration=4π,step=π/16,remeasure=true)
sim_gif!(block(amp=π/4);duration=8π,step=π/16,remeasure=true,μbody=true,cfill=:Blues,legend=false,border=:none)


function foil(L=2^6; thk=2+√2, U=1, ν=1e-4)
    # fraction along length
    s(x) = clamp(x/L,0,1)
    
    # thickened line
    sdf(x,t) = √sum(abs2,x-L*SVector(s(x[1]),0.))-0.5thk

    # displacement
    map(x,t) = SVector(x[1]-L, x[2]-L-L*A(t*U/L)*ψ(s(x[1]-L)))
    
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

	# plot the vorcity ω=curl(u) scaled by the body length L and flow speed U
	function plot_vorticity(sim)
		@inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
		contourf(sim.flow.σ', 
			color=palette(:BuGn), clims=(-10,10),linewidth=0,
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


    using Symbolics
    @variables R, a, θ,  t, β(t)

    z = a + R*(cos(θ) + im*sin(θ)) - R*(cos(β) - im*sin(β))
    x = real(z + 1/z)
    y = imag(z+1/z)

    Differential(x,t)
    D = Differential(t)
    expand_derivatives(D(x))
    D(y)
    D(β)