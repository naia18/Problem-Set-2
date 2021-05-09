# MAIN LIBRARIES to use in Ex. 2
using Optim
using Statistics
using Plots
using LinearAlgebra
using DataFrames
using StatsFuns

function lag(x::Array{Float64,2},p::Int64)
	n,k = size(x)
	lagged_x = [ones(p,k); x[1:n-p,:]]
end

function lag(x::Array{Float64,1},p::Int64)
	n = size(x,1)
	lagged_x = [ones(p); x[1:n-p]]
end	 


function  lags(x::Array{Float64,2},p)
	n, k = size(x)
	lagged_x = zeros(eltype(x),n,p*k)
	for i = 1:p
		lagged_x[:,i*k-k+1:i*k] = lag(x,i)
	end
    return lagged_x
end	

function  lags(x::Array{Float64,1},p)
	n = size(x,1)
	lagged_x = zeros(eltype(x), n,p)
	for i = 1:p
		lagged_x[:,i] = lag(x,i)
	end
    return lagged_x
end

# ------------------- MAIN FUNCTIONS ------------------------- 

# what is the best moment to use???

# moment condition
function GIVmoments(theta, data)
    data = [data lags(data,2)]
    data = data[3:end,:] # get rid of missings
    n = size(data,1)
    y = data[:,1]
    ylag = data[:,2]
    x = data[:,3]
    xlag = data[:,6]
    xlag2 = data[:,9]
    X = [ones(n,1) ylag x]
    e = y - X*theta
    Z = [ones(n,1) x xlag xlag2]
    m = e.*Z
    return m
end

function gmm(moments, theta, data, weight)
    # average moments
    m = theta -> vec(mean(moments(theta,data),dims=1)) # 1Xg   
    # GMM objective function
    obj = theta -> ((m(theta))'weight*m(theta))
    # Minimization
    thetahat, objvalue, converged = fminunc(obj, theta)
    # moment contributions at estimate
    mc_thetahat = moments(thetahat,data)
    return thetahat, objvalue, mc_thetahat, converged
end

# Unconstrained minimization problem    
function fminunc(obj, x; tol = 1e-08)
    results = Optim.optimize(obj, x, LBFGS(), 
                            Optim.Options(
                            g_tol = tol,
                            x_tol=tol,
                            f_tol=tol))
    return results.minimizer, results.minimum, Optim.converged(results)
    #xopt, objvalue, flag = fmincon(obj, x, tol=tol)
    #return xopt, objvalue, flag
end

# ------------------- SET UP ------------------------- 

n = 1000
x = randn(n) # an exogenous regressor
e = randn(n) # the error term
ystar = zeros(n)
alpha_0 = 0.0
rho_0 = 0.9
beta_0 = 1.0
theta = [alpha_0 rho_0 beta_0]
# generate the unobserved dependent variable
for t = 2:n
  ystar[t] = theta[1] + theta[2]*ystar[t-1] + theta[3]*x[t] + e[t]
end

# generate the observed dependent variable by adding measurement error
sig = 1
y = ystar + sig*randn(n);
ylag = lag(y,1);
data = [y ylag x];
data = data[2:end,:]; # drop first observation, missing due to lag

# ------------------- GMM TWO-STEP ESTIMATION ------------------------- 

theta_trial = [1.0, 0.5, 0.5]                     # Trial value of parameter estimators
moments = (theta,data) -> GIVmoments(theta,data)  # Generate the moments function to send as an argument for the gmm()

# -------------- FIRST ESTIMATION ---------------
thetahat1, objval, ms, converged = gmm(moments, theta_trial, data, I(4));
W = inv(cov(ms)); 
# use  thetahat1 to re-estimate by defining a specific weighting matrix

# -------------- SECOND ESTIMATION --------------
thetahat, objval, ms, converged = gmm(moments, thetahat1, data, W);

# Compare the estimators
df = DataFrame(Thetas = ["True value θ_0", "First estimation θ1_hat", "Second estimation θ2_hat"], Values = [theta, thetahat1, thetahat])


