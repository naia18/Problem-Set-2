# MAIN LIBRARIES to use
using Optim
using Statistics
using LinearAlgebra
using CSV
using DataFrames
using StatsFuns

data = DataFrame(CSV.File("nerlove.csv"))
data = log.(data[:,[:cost,:output,:labor,:fuel,:capital]]);

n = size(data,1)
y = data[:,1]
x = data[:,2:end]
x[!,:intercept]=ones(size(data,1))
x = x[!,[:intercept,:output,:labor,:fuel,:capital]];   # add a column "intercept" of ones at index 1 

# Create X and y
y = convert(Array,y)                                   # turn both DataFrames into Arrays
x = convert(Array,x);

function ols(y::Array{Float64}, x::Array{Float64,2}; R=[], r=[], vc="white", silent=false)
        
    # compute ols coefficients, fitted values, and errors
    function lsfit(y, x)
        beta = inv(x'*x)*x'*y
        fit = x*beta
        errors = y - fit
        return beta, fit, errors
    end

    n,k = size(x)
    b, fit, e = lsfit(y,x)
    df = n-k
    sigsq = (e'*e/df)[1,1]
    xx_inv = inv(x'*x)
    ess = (e' * e)[1,1]
    
    # Restricted LS
    if R !=[]
        res_flag = true      # Restricted_flag True if restrictions
        q = size(R,1)
        P_inv = inv(R*xx_inv*R')
        b = b .- xx_inv*R'*P_inv*(R*b.-r)
        e = y-x*b;
        ess = (e' * e)[1,1]
        df = n-k-q
        sigsq = ess/df
        A = Matrix{Float64}(I, k, k) .- xx_inv*R'*P_inv*R;  # the matrix relating b and b_r
    end

    xe = x.*e
    varb = xx_inv*xe'xe*xx_inv

    # Restricted LS
    if R !=[]
        varb = A*varb*A'
    end

    # We only need the SE
    seb = sqrt.(diag(varb))
    seb = seb.*(seb.>1e-16) # round off to zero when there are restrictions

    return b, seb, res_flag
end

# Set restrictions: 
R = [0 1 0 0 0]        # CRTS if \beta_q=1
r = 1;

# ----------------- OLS ESTIMATION -----------------------

(b, seb, flg) = ols(y, x, R=R, r=r);

# Print results
if flg
    print("Restricted LS:\n\n")
    df = DataFrame(Estimators = ["beta_hat1", "beta_hat2", "beta_hat3", "beta_hat4", "beta_hat5"], Values = b, Standard_Errors=seb)
else
    print("Non-restricted LS:\n\n")
    df = DataFrame(Estimators = ["beta_hat1", "beta_hat2", "beta_hat3", "beta_hat4", "beta_hat5"], Values = b, Standard_Errors=seb)
end



function TestStatistics(y, x, R, r; silent=false)
    n,k = size(x)
    q = size(R,1)
    b = x\y
    xx_inv = inv(x'*x)
    P_inv = inv(R*xx_inv*R')
    b_r = b .- xx_inv*R'*P_inv*(R*b.-r)
    e = y - x*b
    ess = (e'*e)[1]
    e_r = y - x*b_r
    ess_r = (e_r' * e_r)[1]
    sigsqhat = ess/(n)
    sigsqhat_r = ess_r/(n)
    # Wald test (uses unrestricted model's est. of sig^2)
    W = (R*b.-r)'*P_inv*(R*b.-r)/sigsqhat
    # LR test
    lnl = -n/2*log(2*pi) - n/2*log(sigsqhat) - ess/(2.0*sigsqhat)
    lnl_r = -n/2*log(2*pi) - n/2*log(sigsqhat_r) - ess_r/(2.0*sigsqhat_r)
    LR = 2.0*(lnl-lnl_r)
    # Score test (uses restricted model's est. of sig^2)
    P_x = x * xx_inv * x'
    S = e_r' * P_x * e_r/(sigsqhat_r)
    
    tests_label = ["Wald","LR","LM"]
    tests = [W[1], LR[1], S[1]]
    pvalues = chisqccdf.(q,tests)
    
    return tests_label, tests, pvalues
end

# -------------- TESTING --------------
t_label, tests, pval = TestStatistics(y, x, R, r)
df = DataFrame(Test_type = t_label, Test_values = tests, p_values = pval)


