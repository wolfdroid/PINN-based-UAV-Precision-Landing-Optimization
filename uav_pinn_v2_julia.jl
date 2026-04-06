#=
=============================================================================
  UAV Drone Precision Landing — PINN Control Strategy v2.0 (Julia / Flux.jl)
  Multi-threaded CPU implementation for benchmarking against PyTorch + CUDA.
=============================================================================

Requirements (install once):
=#
using Pkg
Pkg.add(["Flux", "Zygote", "Optimisers", "ForwardDiff", "Optim", "LineSearches",
            "Plots", "XLSX", "Random", "LinearAlgebra", "Statistics", "Printf"])
#=
Usage (IMPORTANT — set threads BEFORE Julia starts):
    julia -t auto uav_pinn_v2_julia.jl                # Use all cores
    julia -t 12   uav_pinn_v2_julia.jl                # Use 12 threads
    julia -t auto uav_pinn_v2_julia.jl --epochs=8000

Threading strategy:
  - BLAS threads set to 3/4 of available Julia threads (Dense layer matmuls)
  - Parallel batch generation via @threads
  - Parallel RK4 verification across scenarios via @threads
  - JIT warmup before timing

All 15 changes identical to PyTorch version.
=#

using Flux
using Zygote
using Optimisers
using ForwardDiff
using Optim
using LineSearches
using Random
using LinearAlgebra
using Statistics
using Printf
using Plots; gr()
using XLSX
using Base.Threads: @threads, nthreads, threadid

# ═══════════════════════════════════════════════════════════════════════════
#  0. THREAD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

function configure_threads!()
    n_julia = nthreads()
    n_blas = max(1, div(3 * n_julia, 4))  # 3/4 of available threads for BLAS
    BLAS.set_num_threads(n_blas)
    println("  Julia threads: $n_julia")
    println("  BLAS threads:  $n_blas (3/4 of available)")
    return n_julia, n_blas
end

# ═══════════════════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

Base.@kwdef struct UAVParams
    m::Float64=1.5; g::Float64=9.81
    k_dx::Float64=0.25; k_dy::Float64=0.25; k_dz::Float64=0.30
    z0::Float64=10.0; yaw::Float64=0.0
    theta_max_deg::Float64=30.0; phi_max_deg::Float64=30.0
    thrust_min_ratio::Float64=0.5; thrust_max_ratio::Float64=2.0
    T_min::Float64=2.0; T_max::Float64=8.0
    enable_ground_effect::Bool=false; R_rotor::Float64=0.25
end

Base.@kwdef struct TrainConfig
    adam_epochs::Int=4000; lbfgs_epochs::Int=500
    batch_size::Int=512; n_colloc::Int=80
    lr::Float64=1e-3; print_every::Int=500
    target_xy_range::Float64=8.0; wind_range::Float64=3.0
    w_pde::Float64=2.0; w_ic::Float64=40.0; w_bc::Float64=40.0
    w_ground::Float64=10.0; w_vel_hz::Float64=15.0
end

# ═══════════════════════════════════════════════════════════════════════════
#  2. PINN NETWORK
# ═══════════════════════════════════════════════════════════════════════════

struct PINNModel
    fc0::Dense; fc1::Dense; fc2::Dense; fc3::Dense; fc_out::Dense
    theta_max::Float64; phi_max::Float64
    F_min::Float64; F_max::Float64
    xy_scale::Float64; z_scale::Float64; v_scale::Float64
    T_min::Float64; T_max::Float64
end
Flux.@layer PINNModel

function PINNModel(params::UAVParams, cfg::TrainConfig; hidden::Int=256)
    PINNModel(
        Dense(5=>hidden, tanh; init=Flux.glorot_normal),
        Dense(hidden=>hidden, tanh; init=Flux.glorot_normal),
        Dense(hidden=>hidden, tanh; init=Flux.glorot_normal),
        Dense(hidden=>hidden, tanh; init=Flux.glorot_normal),
        Dense(hidden=>10; init=Flux.glorot_normal),
        deg2rad(params.theta_max_deg), deg2rad(params.phi_max_deg),
        params.thrust_min_ratio*params.m*params.g,
        params.thrust_max_ratio*params.m*params.g,
        cfg.target_xy_range*1.5, params.z0, cfg.target_xy_range,
        params.T_min, params.T_max)
end

function (m::PINNModel)(tau, task)
    inp = vcat(tau, task)
    h = m.fc0(inp); h1 = m.fc1(h)
    h2 = m.fc2(h1) .+ h1; h3 = m.fc3(h2) .+ h2
    raw = m.fc_out(h3)
    states = vcat(raw[1:1,:].*m.xy_scale, raw[2:2,:].*m.xy_scale,
                  raw[3:3,:].*m.z_scale, raw[4:4,:].*m.v_scale,
                  raw[5:5,:].*m.v_scale, raw[6:6,:].*m.v_scale)
    ctrls = vcat(m.theta_max.*tanh.(raw[7:7,:]), m.phi_max.*tanh.(raw[8:8,:]),
                 m.F_min.+(m.F_max-m.F_min).*Flux.sigmoid.(raw[9:9,:]))
    T = m.T_min .+ (m.T_max-m.T_min).*Flux.sigmoid.(raw[10:10,:])
    return states, ctrls, T
end

# ═══════════════════════════════════════════════════════════════════════════
#  3. PINN LOSS
# ═══════════════════════════════════════════════════════════════════════════

function compute_pinn_loss(model::PINNModel, task_batch::Matrix{Float32},
                           cfg::TrainConfig, params::UAVParams)
    B = size(task_batch, 2); N = cfg.n_colloc
    tau_c = rand(Float32, 1, B*N)
    task_c = repeat(task_batch, 1, N)
    states, ctrls, T_pred = model(tau_c, task_c)
    # Central finite differences for dS/dτ — O(ε²) accuracy [13]
    # ε=1e-4 gives ~1e-8 truncation error (vs. ~5e-9 for forward FD with ε=5e-5)
    ε = 1f-4
    states_p, _, _ = model(tau_c .+ ε, task_c)
    states_m, _, _ = model(tau_c .- ε, task_c)
    dS = (states_p .- states_m) ./ (2f0 * ε)
    vx=states[4:4,:]; vy=states[5:5,:]; vz=states[6:6,:]; z=states[3:3,:]
    θ=ctrls[1:1,:]; ϕ=ctrls[2:2,:]; F=ctrls[3:3,:]
    vwx=task_c[3:3,:]; vwy=task_c[4:4,:]
    # Ground effect [6]
    if params.enable_ground_effect
        z_safe = max.(z, 0.1f0)
        ge = 1f0 .+ (Float32(params.R_rotor) ./ (4f0 .* z_safe)).^2
        F_eff = F .* ge
    else
        F_eff = F
    end
    ax=(F_eff.*sin.(θ).*cos.(ϕ).-params.k_dx.*(vx.-vwx))./params.m
    ay=(.-F_eff.*sin.(ϕ).-params.k_dy.*(vy.-vwy))./params.m
    az=(F_eff.*cos.(θ).*cos.(ϕ).-params.m*params.g.-params.k_dz.*vz)./params.m
    f_phys = vcat(vx,vy,vz,ax,ay,az)
    loss_pde = mean((dS .- T_pred.*f_phys).^2)
    loss_ground = mean(relu.(.-z).^2)
    s0,_,_ = model(zeros(Float32,1,B), task_batch)
    loss_ic = mean((s0 .- Float32[0,0,params.z0,0,0,0]).^2)
    s1,_,_ = model(ones(Float32,1,B), task_batch)
    loss_bc = mean((s1[1:2,:].-task_batch[1:2,:]).^2) +
              mean(s1[3:3,:].^2)*3f0 + mean(s1[6:6,:].^2)*2f0 +
              cfg.w_vel_hz*mean(s1[4:5,:].^2)
    total = cfg.w_pde*loss_pde + cfg.w_ic*loss_ic + cfg.w_bc*loss_bc + cfg.w_ground*loss_ground
    return total, loss_pde, loss_ic, loss_bc, loss_ground
end

# ═══════════════════════════════════════════════════════════════════════════
#  4. THREADED BATCH & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

function sample_task_threaded(cfg::TrainConfig)
    task = Matrix{Float32}(undef, 4, cfg.batch_size)
    cs = max(1, div(cfg.batch_size, nthreads()))
    @threads for tid in 1:nthreads()
        lo = (tid-1)*cs+1; hi = tid==nthreads() ? cfg.batch_size : tid*cs
        lo > cfg.batch_size && continue; nc = hi-lo+1
        task[1:2,lo:hi] .= (rand(Float32,2,nc).-0.5f0).*2f0.*cfg.target_xy_range
        task[3:4,lo:hi] .= (rand(Float32,2,nc).-0.5f0).*2f0.*cfg.wind_range
    end
    return task
end

cosine_lr(ep,mx,lr0,lr_min=1e-5) = lr_min+0.5*(lr0-lr_min)*(1.0+cos(π*ep/mx))
get_ram_mb() = try round(Sys.maxrss()/1e6;digits=1) catch; 0.0 end

function get_cpu_seconds()
    try
        Sys.islinux() || return 0.0
        fields = split(read("/proc/self/stat", String))
        return (parse(Float64,fields[14])+parse(Float64,fields[15])) / 100.0
    catch; 0.0 end
end

# ═══════════════════════════════════════════════════════════════════════════
#  5. TRAINING
# ═══════════════════════════════════════════════════════════════════════════

function train_model(model::PINNModel, params::UAVParams, cfg::TrainConfig)
    history = Dict("epoch"=>Int[],"pde"=>Float64[],"ic"=>Float64[],
                   "bc"=>Float64[],"ground"=>Float64[],"total"=>Float64[])
    prof = Dict("epoch"=>Int[],"phase"=>String[],"wall_time_s"=>Float64[],
        "epoch_time_ms"=>Float64[],"loss_total"=>Float64[],"loss_pde"=>Float64[],
        "loss_ic"=>Float64[],"loss_bc"=>Float64[],"loss_ground"=>Float64[],
        "learning_rate"=>Float64[],"cpu_percent"=>Float64[],"ram_used_mb"=>Float64[],
        "ram_percent"=>Float64[],"gpu_mem_allocated_mb"=>Float64[],
        "gpu_mem_reserved_mb"=>Float64[],"gpu_utilization_pct"=>Float64[])

    t_start = 0.0

    function record!(ep,ph,t0,tot,pde,ic,bc,gnd,lr)
        now=time()
        push!(prof["epoch"],ep); push!(prof["phase"],ph)
        push!(prof["wall_time_s"],round(now-t_start;digits=3))
        push!(prof["epoch_time_ms"],round((now-t0)*1000;digits=2))
        push!(prof["loss_total"],Float64(tot)); push!(prof["loss_pde"],Float64(pde))
        push!(prof["loss_ic"],Float64(ic)); push!(prof["loss_bc"],Float64(bc))
        push!(prof["loss_ground"],Float64(gnd)); push!(prof["learning_rate"],Float64(lr))
        push!(prof["ram_used_mb"],get_ram_mb()); push!(prof["cpu_percent"],get_cpu_seconds())
        push!(prof["ram_percent"],0.0)
        push!(prof["gpu_mem_allocated_mb"],0.0); push!(prof["gpu_mem_reserved_mb"],0.0)
        push!(prof["gpu_utilization_pct"],0.0)
    end

    opt_state = Optimisers.setup(Optimisers.Adam(cfg.lr), model)

    println("  Warming up JIT...")
    wt = sample_task_threaded(cfg)
    Flux.withgradient(model) do m; compute_pinn_loss(m,wt,cfg,params) end
    println("  JIT warmup complete.\n")

    println("="^70)
    println("  Phase 1: Adam ($(nthreads()) Julia thr, $(BLAS.get_num_threads()) BLAS thr)")
    println("="^70)

    t_start = time()
    for epoch in 1:cfg.adam_epochs
        et0 = time()
        task = sample_task_threaded(cfg)
        (tot,pde,ic,bc,gnd),grads = Flux.withgradient(model) do m
            compute_pinn_loss(m,task,cfg,params)
        end
        gc = Flux.fmap(grads[1]) do g
            g===nothing && return nothing
            gn=norm(g); gn>5f0 ? g.*(5f0/gn) : g
        end
        lr_now = cosine_lr(epoch, cfg.adam_epochs, cfg.lr)
        Optimisers.adjust!(opt_state, lr_now)
        opt_state, model = Optimisers.update!(opt_state, model, gc)
        record!(epoch,"adam",et0,tot,pde,ic,bc,gnd,lr_now)
        if epoch%cfg.print_every==0 || epoch==1
            el=time()-t_start
            @printf("  Ep %5d | Tot: %9.4f | PDE: %.4f | IC: %.4f | BC: %.4f | Gnd: %.6f | LR: %.6f | %.1fs\n",
                    epoch,tot,pde,ic,bc,gnd,lr_now,el)
            for (k,v) in [("epoch",epoch),("total",Float64(tot)),("pde",Float64(pde)),
                          ("ic",Float64(ic)),("bc",Float64(bc)),("ground",Float64(gnd))]
                push!(history[k], v)
            end
        end
    end
    @printf("\n  Adam complete in %.1fs\n\n", time()-t_start)

    println("="^70)
    println("  Phase 2: L-BFGS Precision Tuning")
    println("="^70)
    t2=time()
    lbfgs_macro_steps = cfg.lbfgs_epochs ÷ 20
    task_l = sample_task_threaded(cfg)

    for step in 1:lbfgs_macro_steps
        et0 = time()
        ce = cfg.adam_epochs + step * 20

        if step % 3 == 1
            task_l = sample_task_threaded(cfg)
        end

        # Flatten model parameters for Optim.jl via Flux.destructure
        θ_flat, re = Flux.destructure(model)
        θ0 = Float64.(θ_flat)

        # Capture current task_l into a local so the closure does not hold a
        # reference to the outer mutable variable (Julia loop-closure semantics).
        task_step = task_l

        function loss_flat(θ64::Vector{Float64})
            m = re(Float32.(θ64))
            total, = compute_pinn_loss(m, task_step, cfg, params)
            return Float64(total)
        end

        function fg!(F, G, θ)
            if G !== nothing
                val, back = Zygote.pullback(loss_flat, θ)
                G .= back(1.0)[1]
                F !== nothing && return val
            elseif F !== nothing
                return loss_flat(θ)
            end
        end

        result = Optim.optimize(
            loss_flat,
            (G, θ) -> begin
                _, back = Zygote.pullback(loss_flat, θ)
                G .= back(1.0)[1]
            end,
            θ0,
            Optim.LBFGS(m=50, linesearch=LineSearches.BackTracking(order=3)),
            Optim.Options(iterations=20, show_trace=false)
        )

        model = re(Float32.(Optim.minimizer(result)))

        eval_task = sample_task_threaded(cfg)
        tot, pde, ic, bc, gnd = compute_pinn_loss(model, eval_task, cfg, params)
        record!(ce, "lbfgs", et0, tot, pde, ic, bc, gnd, 0.0)

        if step % 5 == 0 || step == 1
            @printf("  L-BFGS step %4d (ep ~%d) | Tot: %9.4f | PDE: %.4f | IC: %.4f | BC: %.4f | %.1fs\n",
                    step, ce, tot, pde, ic, bc, time()-t2)
            for (k,v) in [("epoch",ce),("total",Float64(tot)),("pde",Float64(pde)),
                          ("ic",Float64(ic)),("bc",Float64(bc)),("ground",Float64(gnd))]
                push!(history[k], v)
            end
        end
    end
    @printf("\n  Phase 2 in %.1fs | Total: %.1fs\n\n", time()-t2, time()-t_start)
    return model, history, prof
end

# ═══════════════════════════════════════════════════════════════════════════
#  6. RK4 VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

function rk4_verify(model::PINNModel, params::UAVParams,
                    target::Vector{Float64}, wind::Vector{Float64}; n_steps::Int=500)
    ta = Float32[target[1],target[2],wind[1],wind[2]]
    t1 = reshape(ta,4,1)
    _,_,Tp = model(Float32[0.5;;],t1); Tv=Float64(Tp[1])
    dt=Tv/n_steps; st=[0.0,0.0,params.z0,0.0,0.0,0.0]
    traj = Vector{Vector{Float64}}(undef,n_steps+1); traj[1]=copy(st)
    function dyn(s,tv)
        _,c,_=model(Float32[tv;;],t1); c=Float64.(c[:,1])
        vx,vy,vz=s[4],s[5],s[6]
        ax=(c[3]*sin(c[1])*cos(c[2])-params.k_dx*(vx-wind[1]))/params.m
        ay=(-c[3]*sin(c[2])-params.k_dy*(vy-wind[2]))/params.m
        az=(c[3]*cos(c[1])*cos(c[2])-params.m*params.g-params.k_dz*vz)/params.m
        [vx,vy,vz,ax,ay,az]
    end
    for i in 0:n_steps-1
        tv=Float32(i/n_steps); dτ=Float32(1/n_steps)
        k1=dyn(st,tv); k2=dyn(st.+0.5.*dt.*k1,tv+0.5f0*dτ)
        k3=dyn(st.+0.5.*dt.*k2,tv+0.5f0*dτ); k4=dyn(st.+dt.*k3,tv+dτ)
        st=st.+(dt/6).*(k1.+2 .*k2.+2 .*k3.+k4); traj[i+2]=copy(st)
    end
    rm=reduce(hcat,traj)'
    tm=reshape(Float32.(range(0,1;length=n_steps+1)),1,n_steps+1)
    ps,pc,_=model(tm,repeat(t1,1,n_steps+1))
    pt=Float64.(ps'); ct=Float64.(pc')
    pe=sqrt.(sum((rm[:,1:3].-pt[:,1:3]).^2;dims=2))[:,1]
    Dict("rk4_traj"=>rm,"pinn_traj"=>pt,"pinn_ctrls"=>ct,"T"=>Tv,
         "pos_error"=>pe,"max_error"=>maximum(pe),"mean_error"=>mean(pe),
         "final_rk4"=>rm[end,:],"final_pinn"=>pt[end,:])
end

function rk4_verify_all(model, params, scenarios)
    results = Vector{Dict}(undef, length(scenarios))
    @threads for i in eachindex(scenarios)
        results[i] = rk4_verify(model, params, scenarios[i].target, scenarios[i].wind)
    end
    results
end

# ═══════════════════════════════════════════════════════════════════════════
#  7. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

const C_BG=RGB(10/255,10/255,15/255); const C_TEXT=RGB(216/255,216/255,220/255)
const C_RED1=RGB(255/255,45/255,45/255); const C_RED3=RGB(255/255,107/255,107/255)
const C_SILVER=RGB(168/255,176/255,188/255); const C_STEEL=RGB(107/255,122/255,141/255)
const C_GOLD=RGB(201/255,168/255,76/255); const C_GRID=RGB(30/255,32/255,48/255)
const C_PANEL=RGB(18/255,19/255,26/255)

function evaluate_and_visualize(model::PINNModel, params::UAVParams,
                                cfg::TrainConfig, history::Dict)
    scenarios = [(target=[10.0,-5.0],wind=[2.0,-1.0],label="Crosswind"),
                 (target=[-6.0,4.0],wind=[-1.5,2.0],label="Reverse Wind"),
                 (target=[3.0,3.0],wind=[0.0,0.0],label="No Wind"),
                 (target=[8.0,8.0],wind=[3.0,3.0],label="Strong Diag.")]

    println("  Running RK4 verification ($(nthreads()) threads)...")
    ar = rk4_verify_all(model, params, scenarios)

    println("="^70); println("  EVALUATION RESULTS"); println("="^70)
    eval_results = []
    for (i,(sc,r)) in enumerate(zip(scenarios,ar))
        f=r["final_pinn"]; t=sc.target
        pe=sqrt((f[1]-t[1])^2+(f[2]-t[2])^2+f[3]^2)
        ve=sqrt(f[4]^2+f[5]^2+f[6]^2)
        @printf("\n  Scenario %d: %s\n",i,sc.label)
        @printf("    Target: [%.1f,%.1f] Wind: [%.1f,%.1f]\n",t[1],t[2],sc.wind[1],sc.wind[2])
        @printf("    Final: [%.3f,%.3f,%.3f] PosErr: %.4f VelErr: %.4f T: %.2f\n",
                f[1],f[2],f[3],pe,ve,r["T"])
        @printf("    RK4 max: %.4f mean: %.4f\n",r["max_error"],r["mean_error"])
        push!(eval_results, Dict("label"=>sc.label,"target"=>sc.target,"wind"=>sc.wind,
            "final_pos"=>f[1:3],"final_vel"=>f[4:6],"pos_error"=>pe,"vel_error"=>ve,
            "T"=>r["T"],"rk4_max_dev"=>r["max_error"],"rk4_mean_dev"=>r["mean_error"]))
    end
    println("="^70)

    r1=ar[1]; Tv=r1["T"]; npt=size(r1["pinn_traj"],1)
    ta=range(0,Tv;length=npt); ps=r1["pinn_traj"]; pc=r1["pinn_ctrls"]; rs=r1["rk4_traj"]

    # Panel 1: 3D — metallic red + silver
    p1=plot3d(ps[:,1],ps[:,2],max.(ps[:,3],0);label="PINN",lw=2.5,
        color=C_RED1, xlabel="X(m)",ylabel="Y(m)",zlabel="Z(m)",
        title="3D Trajectory + RK4",bg=C_BG,fg=C_TEXT,
        guidefontcolor=C_STEEL, tickfontcolor=C_STEEL,
        legend=:topleft,legendfontcolor=C_TEXT,legendbg=C_PANEL)
    plot3d!(rs[:,1],rs[:,2],max.(rs[:,3],0);label="RK4",lw=1.5,ls=:dash,
        color=C_SILVER,alpha=0.7)
    scatter3d!([0],[0],[params.z0];label="Start",color=C_GOLD,ms=6)
    scatter3d!([10.0],[-5.0],[0];label="Target",color=C_RED3,ms=8,shape=:xcross)

    # Panel 2: Velocity — red/silver/gold for 3 components
    p2=plot(ta,ps[:,4];label="vₓ",lw=2,color=C_RED1,xlabel="Time (s)",
        ylabel="Velocity (m/s)",title="Velocity Profile",bg=C_BG,fg=C_TEXT,
        guidefontcolor=C_STEEL,tickfontcolor=C_STEEL,
        legend=:topright,legendfontcolor=C_TEXT,legendbg=C_PANEL)
    plot!(ta,ps[:,5];label="vᵧ",lw=2,color=C_SILVER)
    plot!(ta,ps[:,6];label="v_z",lw=2,color=C_GOLD)
    hline!([0];label=false,ls=:dash,color=C_STEEL,alpha=0.3)

    # Panel 3: Controls — thrust red, pitch gold (no twinx overlap issue)
    p3=plot(ta,pc[:,3];label="Thrust F (N)",lw=2.5,color=C_RED1,
        xlabel="Time (s)",ylabel="Thrust (N)",title="Control Commands",
        bg=C_BG,fg=C_TEXT,guidefontcolor=C_STEEL,tickfontcolor=C_STEEL,
        legend=:topleft,legendfontcolor=C_TEXT,legendbg=C_PANEL)
    p3r=twinx()
    plot!(p3r,ta,rad2deg.(pc[:,1]);label="Pitch θ (°)",lw=2,ls=:dashdot,
        color=C_GOLD,ylabel="Angle (°)",guidefontcolor=C_STEEL,
        tickfontcolor=C_STEEL,legend=:bottomright,legendfontcolor=C_TEXT,
        legendbg=C_PANEL)

    # Panel 4: Altitude — red fill, silver RK4
    p4=plot(ta,ps[:,3];label="PINN z",lw=2.5,color=C_RED1,
        xlabel="Time (s)",ylabel="Z (m)",title="Altitude & Ground Safety",
        bg=C_BG,fg=C_TEXT,guidefontcolor=C_STEEL,tickfontcolor=C_STEEL,
        fill=(0,0.08,C_RED1),legend=:topright,legendfontcolor=C_TEXT,
        legendbg=C_PANEL)
    plot!(ta,rs[:,3];label="RK4 z",lw=1.5,ls=:dash,color=C_SILVER,alpha=0.7)
    hline!([0];label="Ground",color=C_RED3,lw=1.5,alpha=0.5)

    # Panel 5: Loss curves — red/gold/silver/salmon
    ep=history["epoch"]
    p5=plot(ep,max.(history["pde"],1e-12);label="PDE",lw=2,color=C_RED1,
        yscale=:log10,xlabel="Epoch",ylabel="Loss (log)",
        title="Training Loss Curves",bg=C_BG,fg=C_TEXT,
        guidefontcolor=C_STEEL,tickfontcolor=C_STEEL,
        legend=:topright,legendfontcolor=C_TEXT,legendbg=C_PANEL)
    plot!(ep,max.(history["ic"],1e-12);label="IC",lw=2,color=C_GOLD)
    plot!(ep,max.(history["bc"],1e-12);label="BC",lw=2,color=C_SILVER)
    plot!(ep,max.(history["ground"],1e-12);label="Ground",lw=2,color=C_RED3)

    # Panel 6: Multi-scenario bars — red + silver, no overlap
    pe6=[ev["pos_error"] for ev in eval_results]
    ve6=[ev["vel_error"] for ev in eval_results]
    x6=1:4; nms=[s.label for s in scenarios]
    p6=bar(x6 .- 0.2, pe6; bar_width=0.3, label="Pos (m)", color=C_RED1,
        linecolor=C_RED3, xticks=(x6,nms), xrotation=20, title="Multi-Scenario Accuracy",
        bg=C_BG, fg=C_TEXT, guidefontcolor=C_STEEL, tickfontcolor=C_STEEL,
        legend=:topleft, legendfontcolor=C_TEXT, legendbg=C_PANEL, ylabel="Error")
    bar!(x6 .+ 0.2, ve6; bar_width=0.3, label="Vel (m/s)", color=C_SILVER,
        linecolor=C_STEEL)

    # Compose dashboard
    dash=plot(p1,p2,p3,p4,p5,p6;layout=(2,3),size=(2000,1200),
        plot_title="UAV PINN — Julia ($(nthreads()) threads)",
        plot_titlefontcolor=C_RED1, plot_titlefontsize=16,
        bg=C_BG,fg=C_TEXT, margin=8Plots.mm)
    savefig(dash,"pinn_uav_dashboard_julia.png")
    println("\n  Dashboard -> pinn_uav_dashboard_julia.png")

    # Multi-scenario 3D
    clr=[C_RED1, C_SILVER, C_GOLD, C_RED3]
    pm=plot3d(;xlabel="X(m)",ylabel="Y(m)",zlabel="Z(m)",
        title="Multi-Scenario Trajectories",bg=C_BG,fg=C_TEXT,
        guidefontcolor=C_STEEL,tickfontcolor=C_STEEL,
        size=(900,750),legend=:topleft,legendfontcolor=C_TEXT,legendbg=C_PANEL)
    for (i,(sc,c)) in enumerate(zip(scenarios,clr))
        tr=ar[i]["pinn_traj"]
        plot3d!(tr[:,1],tr[:,2],max.(tr[:,3],0);label=sc.label,lw=2.5,color=c)
        scatter3d!([sc.target[1]],[sc.target[2]],[0];label=false,color=c,ms=6,shape=:xcross)
    end
    scatter3d!([0],[0],[params.z0];label="Start",color=C_GOLD,ms=8)
    savefig(pm,"pinn_multi_trajectory_julia.png")
    println("  Multi-scenario -> pinn_multi_trajectory_julia.png\n")
    return eval_results
end

# ═══════════════════════════════════════════════════════════════════════════
#  8. XLSX EXPORT
# ═══════════════════════════════════════════════════════════════════════════

function export_xlsx(prof,hist,evals,params,cfg,npar,njt,nbt;fn="pinn_benchmark_julia.xlsx")
    n=length(prof["epoch"])
    pk=["epoch","phase","wall_time_s","epoch_time_ms","loss_total","loss_pde",
        "loss_ic","loss_bc","loss_ground","learning_rate","cpu_percent","ram_used_mb",
        "ram_percent","gpu_mem_allocated_mb","gpu_mem_reserved_mb","gpu_utilization_pct"]
    ph=["Epoch","Phase","Wall Time(s)","Epoch Time(ms)","Loss Total","Loss PDE",
        "Loss IC","Loss BC","Loss Ground","LR","CPU sec","RAM(MB)","RAM%",
        "GPU Alloc(MB)","GPU Res(MB)","GPU%"]
    XLSX.openxlsx(fn;mode="w") do xf
        s1=XLSX.addsheet!(xf,"Training Profiling")
        for (c,h) in enumerate(ph); s1[1,c]=h end
        for i in 1:n; for (c,k) in enumerate(pk); s1[i+1,c]=prof[k][i] end end

        s2=XLSX.addsheet!(xf,"Summary")
        sd=[["Language","Julia"],["Framework","Flux.jl+Zygote"],["Device","CPU(MT)"],
            ["Julia Version",string(VERSION)],["OS",string(Sys.KERNEL)],
            ["Julia Threads",njt],["BLAS Threads",nbt],["Parameters",npar],
            ["Adam Epochs",cfg.adam_epochs],["L-BFGS Epochs",cfg.lbfgs_epochs],
            ["Batch Size",cfg.batch_size],["Collocation",cfg.n_colloc],
            ["Total Time(s)",n>0 ? prof["wall_time_s"][end] : 0],
            ["Avg Epoch(ms)",n>0 ? round(mean(prof["epoch_time_ms"]);digits=2) : 0],
            ["Med Epoch(ms)",n>0 ? round(median(prof["epoch_time_ms"]);digits=2) : 0],
            ["Final Loss",n>0 ? prof["loss_total"][end] : 0],
            ["Peak RAM(MB)",n>0 ? maximum(prof["ram_used_mb"]) : 0]]
        for (r,row) in enumerate(sd); for (c,v) in enumerate(row); s2[r,c]=v end end

        s3=XLSX.addsheet!(xf,"Evaluation Results")
        eh=["Scenario","TgtX","TgtY","WndVx","WndVy","FinalX","FinalY","FinalZ",
            "FinalVx","FinalVy","FinalVz","PosErr","VelErr","T","RK4Max","RK4Mean"]
        for (c,h) in enumerate(eh); s3[1,c]=h end
        for (r,ev) in enumerate(evals)
            vs=Any[ev["label"],ev["target"][1],ev["target"][2],ev["wind"][1],ev["wind"][2],
                ev["final_pos"][1],ev["final_pos"][2],ev["final_pos"][3],
                ev["final_vel"][1],ev["final_vel"][2],ev["final_vel"][3],
                ev["pos_error"],ev["vel_error"],ev["T"],ev["rk4_max_dev"],ev["rk4_mean_dev"]]
            for (c,v) in enumerate(vs); s3[r+1,c]=v end
        end

        s4=XLSX.addsheet!(xf,"Loss History")
        for (c,h) in enumerate(["Epoch","Total","PDE","IC","BC","Ground"]); s4[1,c]=h end
        for i in 1:length(hist["epoch"])
            s4[i+1,1]=hist["epoch"][i]; s4[i+1,2]=hist["total"][i]
            s4[i+1,3]=hist["pde"][i]; s4[i+1,4]=hist["ic"][i]
            s4[i+1,5]=hist["bc"][i]; s4[i+1,6]=hist["ground"][i]
        end
    end
    println("  XLSX -> $fn")
end

# ═══════════════════════════════════════════════════════════════════════════
#  9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

function fmt(n::Int)
    s=string(n); p=String[]
    while length(s)>3; pushfirst!(p,s[end-2:end]); s=s[1:end-3] end
    pushfirst!(p,s); join(p,",")
end

function main()
    ae=4000; le=500; bs=512; xf="pinn_benchmark_julia.xlsx"; sd=42
    for a in ARGS
        startswith(a,"--epochs=") && (ae=parse(Int,split(a,"=")[2]))
        startswith(a,"--lbfgs-epochs=") && (le=parse(Int,split(a,"=")[2]))
        startswith(a,"--batch-size=") && (bs=parse(Int,split(a,"=")[2]))
        startswith(a,"--xlsx=") && (xf=split(a,"=")[2])
        startswith(a,"--seed=") && (sd=parse(Int,split(a,"=")[2]))
    end
    Random.seed!(sd)
    params=UAVParams(); cfg=TrainConfig(adam_epochs=ae,lbfgs_epochs=le,batch_size=bs)

    println("\n"*"="^70)
    println("  UAV PINN v2.0 — Julia/Flux.jl Multi-Threaded CPU")
    println("="^70)
    njt,nbt = configure_threads!()
    println("  Julia: $VERSION")
    @printf("  Mass:%.1f Drag:(%.2f,%.2f,%.2f) z0:%.1f\n",
            params.m,params.k_dx,params.k_dy,params.k_dz,params.z0)
    @printf("  Thrust:[%.1f,%.1f]N Angles:±%.0f°/±%.0f° T:[%.0f,%.0f]s\n",
            params.thrust_min_ratio*params.m*params.g,
            params.thrust_max_ratio*params.m*params.g,
            params.theta_max_deg,params.phi_max_deg,params.T_min,params.T_max)
    @printf("  Adam(%d)+Fine(%d) Batch:%d Colloc:%d\n",ae,le,bs,cfg.n_colloc)
    println("="^70)

    model = PINNModel(params,cfg)
    np = sum(length, Flux.params(model))
    println("\n  Net: 5→256→256→256→256→10 (skip) | Params: $(fmt(np)) | F32\n")

    model,hist,prof = train_model(model,params,cfg)
    evals = evaluate_and_visualize(model,params,cfg,hist)
    export_xlsx(prof,hist,evals,params,cfg,np,njt,nbt;fn=xf)
    @printf("  Peak RSS: %.1f MB\n", get_ram_mb())
    println("  All done.\n")
end

main()