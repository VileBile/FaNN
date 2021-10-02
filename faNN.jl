using Base: Real
using Plots
using Random
using Shuffle
using Statistics
using MLDatasets: BostonHousing
train_x, train_y = MLDatasets.MNIST.traindata()
features = BostonHousing.features();
target = BostonHousing.targets();
F_min = minimum(features,dims=2)
f_min = F_min
F_d = maximum(features,dims=2) .- F_min
f_d = F_d
T_min = minimum(target)
T_d = maximum(target) .- T_min
target = (target .- T_min) ./ T_d
for i in 1:size(features)[2]-1
    F_min = [F_min f_min]
    F_d = [F_d f_d]   
end
features = (features .- F_min) ./ F_d


#function softmax(Z::Vector{Float64},d=0)
 #   if d == 0 
  #      return (exp.(Z) ./ sum(exp.(Z)) 
   # else

function I(Z,d=0)
   if d == 0 
    return Z
   else 
    return 1 
   end  
end


function sig(z::Real, d = 0)
    if d == 1
        return ((1-sig(z))*sig(z) + 0.1)
    else
        return  1/(1+exp(-z))
    end
end

function swish(z::Real,α=1,d=0)
    if d == 0
        return z/(1+exp(-z)) 
    else 
        return (swish(z,α)/z) + (1 - (swish(z,α)/z)) * swish(z,a)   
    end
end

function relu(Z,d=0)
    if d == 0
        return maximum([0;Z])
    else 
        if Z > 0
            return 1
        else 
            return 0 
        end
    end
end
     

function sqerr(o,t, d = 0)
    if d == 0
        return 0.5*(t-o)^2
    else 
        return o - t 

    end 
end 

   

rng = MersenneTwister(1234);

architecture = Dict{String,Any}("l1"=>(20,sig),"l2"=>=>(20,sig),"l3"=>(20,sig),"l4"=>(20,sig),"l5"=>(1,sig))
a = Dict{String,Any}("l1"=>(5,sig),"l2"=>(1,sig))
function makeNN(a,insize)
    params = Dict{String,Any}
    params = merge!(params, Dict("W1"=>(rand(rng,insize,a["l1"][1] ) .- 0.5),
                                 "af1"=>a["l1"][2],
                                 "b1"=> rand(rng,a["l1"][1])))
    params = merge!(params,Dict("g_W1"=>zeros(size(params["W1"])),
                                "m_W1"=>zeros(size(params["W1"])),
                                "m_b1"=>zeros(size(params["b1"]))))                             
    for i in 2:length(a)
        params = merge!(params, Dict("W" * string(i)=>(rand(rng,a["l"* string(i-1)][1],a["l"* string(i)][1]) .- 0.5),
                                 "af"  * string(i)=>a["l"*string(i)][2],
                                 "b"  * string(i)=>rand(rng,a["l" * string(i)][1])))
        params = merge!(params,Dict("g_W" * string(i)=>zeros(size(params["W" *string(i)])),
                                     "m_W"*string(i)=>zeros(size(params["W"*string(i)])),
                                     "m_b" * string(i)=>zeros(size(params["b"*string(i)]))))

    end
    return params
end

function fp_sl(a_p,W_c,b_c,afun)
    z = transpose(W_c) * a_p + b_c
    return z, afun.(z), afun.(z,1)
end 

 function fp_f(p,x,a)
   memory = Dict{String,Any}("a0" => x) 
   a_c = x
   for i in 1:length(a)
        a_p = a_c
        z_c, a_c, del_a = fp_sl(a_p,p["W"*string(i)],p["b"*string(i)],p["af"*string(i)])
        memory = merge!(memory,Dict("a"*string(i)=>a_c,
                                    "z"*string(i)=>z_c,
                                    "del_a"*string(i) => del_a))
   end
   
   return memory   
end

function gen_cost(o,t)
    err = sqerr.(o,t)
    del_err = sqerr.(o,t,1)
    cost = sum(err)
    return del_err, cost
end

function gen_dels(del_err,p,m,a)
    m = merge!(m,Dict("del_"*string(length(a))=>map(*,m["del_a"*string(length(a))],del_err)))
    for i in length(a)-1:-1:1
    m = merge!(m,Dict("del_"*string(i)=>map(*,(p["W"*string(i+1)]*m["del_"*string(i+1)]),m["del_a"*string(i)])))
    end
    return m
end

function backprop(del_err,p,m,a)
    m = gen_dels(del_err,p,m,a)
    for i in length(a):-1:1
        p["g_W"*string(i)] = m["a"*string(i-1)]*transpose(m["del_"*string(i)])
    end
    return m, p
end


function gd(m,p,a,α,u)
        for i in length(a):-1:1
            db = (α*m["del_"*string(i)] + u*p["m_b"*string(i)])
            dw = (α*p["g_W"*string(i)]  + u*p["m_W"*string(i)]) 
            p["b"*string(i)] = p["b"*string(i)] - db 
            p["W"*string(i)] = p["W"*string(i)] - dw
            p["m_b"*string(i)] = db
            p["m_W"*string(i)] = dw                
        end
        return p
    end


function train(X,Y,a,α,u,epochs = 0, cost_t = 0)
    costs = []
    global P = makeNN(a,size(X)[1])  
    if epochs != 0
        for i in 1:epochs
            c = []
            pop = collect(1:size(X)[2])
            while length(pop) != 0
                indi = splice!(pop,rand(1:length(pop)))   
                global M = fp_f(P,X[:,indi],a)
                del_err, cost = gen_cost(M["a"*string(length(a))],Y[:,indi])
                append!(c,cost)
                global M, P = backprop(del_err,P,M,a)
                global P = gd(M,P,a,α,u)  
        
            end
            append!(costs,mean(c))
            end
        end
        return costs, (P,M ,a)
end



    

