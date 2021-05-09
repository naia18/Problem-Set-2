# MAIN LIBRARIES to use in Ex. 1
using Optim
using Statistics
using Plots
using DataFrames
using StatsFuns
using LaTeXStrings     # To write in Latex typesetting on the plots

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
errors = abs.((theta.-thetahat)./theta)

# results
df = DataFrame(Parameters = ["Theta1", "Theta2"], True_Parameters = theta, ML_estimates=thetahat, Estimation_errors=errors)

# New sample size (larger)
n = 10000
theta = [0.75, 0.75]

# MLE re-optimization
(y,x)=LogitDGP(n, theta)                  # Generate the sample to analyse
obj = theta -> -mean(logL(theta, y, x))   # Define the objective function 
theta_trial = [1.0,0.5]                   # Define a trial value for the parameter to estimate
thetahat, junk, junk = fminunc(obj, theta_trial)
errors = abs.((theta.-thetahat)./theta)

# results
df = DataFrame(Parameters = ["Theta1", "Theta2"], True_Parameters = theta, ML_estimates=thetahat, Estimation_errors=errors)

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
tit = LaTeXString("Histogram of the estimates")
histogram(results,nbins=50,label=[L"\theta_0" L"\theta_1"],title=tit,alpha=0.7)
plot!(size=(500,300))


