# MAIN LIBRARIES to use
using Optim
using Statistics
using ForwardDiff
using Plots
using LinearAlgebra
using CSV
using DataFrames
using StatsFuns

# MAIN FUNCTIONS: Define the Logit Data Generating Process
function LogitDGP(n, theta)
    k = size(theta,1)
    x = ones(n,1)
    if k>1 
        x = [x  randn(n,k-1)]     # create the X matrix
    end
    y = (1.0 ./ (1.0 .+ exp.(-x*theta)) .> rand(n,1))    # compute y values given x and \theta
    return y, x
end

# Define the objective function 
function logL(theta, y, x)
    p = 1.0 ./ (1.0 .+ exp.(-x*theta))
    logdensity = y.*log.(p) .+ (1.0 .- y).*log.(1.0 .- p)
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


# Parameters
n = 1000
theta = [0.75, 0.75]

# MLE
(y,x)=LogitDGP(n, theta)                  # Generate the sample to analyse
obj = theta -> -mean(logL(theta, y, x))   # Define the objective function 
theta_trial = [1.0,0.5]                   # Define a trial value for the parameter to estimate
thetahat, junk, junk = fminunc(obj, theta_trial)

# results
println("the true parameters: ", theta)
println("the ML estimates: ", thetahat)

# New sample size (larger)
n = 10000
theta = [0.75, 0.75]

# MLE re-optimization
(y,x)=LogitDGP(n, theta)                  # Generate the sample to analyse
obj = theta -> -mean(logL(theta, y, x))   # Define the objective function 
theta_trial = [1.0,0.5]                   # Define a trial value for the parameter to estimate
thetahat, junk, junk = fminunc(obj, theta_trial)
errors = (theta.-thetahat)./theta

# results
println("the true parameters: ", theta)
println("the ML estimates: ", thetahat)
println("the error of estimation: ", errors)

n = 300
theta = [0.75, 0.75]
k = size(theta,1)
reps = 1000
results = zeros(reps,k)
for i = 1:reps
    # MLE
    (y,x)=LogitDGP(n, theta)                  # Generate the sample to analyse
    obj = theta -> -mean(logL(theta, y, x))   # Define the objective function 
    theta_trial = [1.0,0.5]                   # Define a trial value for the parameter to estimate   
    thetahat, junk, junk = fminunc(obj, theta_trial)
    results[i,:] = sqrt(n)*((thetahat .- theta)')
end    
histogram(results,nbins=50,label=["theta_0" "theta_1"],title="Histogram of the estimates",alpha=0.7)
plot!(size=(500,300))


