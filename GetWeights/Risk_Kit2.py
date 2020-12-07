import pandas as pd
import numpy as np
import scipy.stats
import math
from scipy.stats import norm
from scipy.optimize import minimize
#import ipywidgets as widgets
#from IPython.display import display
#import matplotlib.pyplot as plt
#import statsmodels.api as sm


def bubs(list1):
    temp=0
    if len(list1)==1:
        return list1
    for count,y in enumerate(list1):
        if  y>list1[0]:
            temp=list1[count]
            list1[count]=list1[0]
            list1[0]=temp
    return [list1[0]]+bubs(list1[1:])


def drawdown(return_series: pd.Series):
    """
    Takes a time series of returns and computes and returns a DataFrame
    that contains Wealth Index, previous peaks and drawdown
    """
    wealth_index=1000*(1+return_series).cumprod()
    previous_peaks=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })


def get_ffme_returns():
    me_m=pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                     header=0, index_col=0, na_values=-99.99)
    rets=me_m[['Lo 10','Hi 10']]
    rets.columns=['Small Cap','Large Cap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_fff_m_returns():
    rets=pd.read_csv("F-F_Research_Data_Factors_m.csv",
                     header=0, index_col=0, na_values=-99.99)
    #rets=me_m[['Lo 10','Hi 10']]
    #rets.columns=['Small Cap','Large Cap']
    rets=rets/100
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
        hfi=pd.read_csv("edhec-hedgefundindices.csv",
                       header=0, index_col=0, parse_dates=True)
        hfi=hfi/100
        hfi.index=hfi.index.to_period('M')
        return hfi

def skewness(r):
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    if (sigma_r**3==0).any():
        print('HIII')
        return 0
    else:
        return exp/sigma_r**3
    

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis(): scipy gives you EXCESS kurtosis
    """
    demeaned_r=r-r.mean()
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    if (sigma_r**4==0).any():
        return 0
    else:
        return exp/sigma_r**4


def is_normal(r, level=.01):
    """
    Use Jarque-Bera test at the X% level.
    """
    statistic, p_value=scipy.stats.jarque_bera(r)
    # result=scipy.stats.jarque_bera(r)
    # return result[1]
    return p_value>level


def is_normal2(r, level=.01):
    """
    Use Jarque-Bera test at the X% level.
    """
    return (r.apply(scipy.stats.jarque_bera, axis=0)).apply(lambda x: x[1]>level)
    
    # result=scipy.stats.jarque_bera(r)
    # return result[1]
    
def semideviation(r):
    return r[r<0].std(ddof=0)

def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return -r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return np.percentile(r, level)
    else:
        raise TypeError("Expected DataFrame or Series")

    
def var_historicApp(r, level=5):
    """
    VaR Historic
    """
    return -r.apply(np.percentile, q=level)


def var_gaussian(r, level=5):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    """
    # Compute the Z-score assuming it is Gaussian
    z=norm.ppf(level/100)
    return -(r.mean()+z*r.std())


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If modified is True, then we adjust it taking into account that r is not normal, but is
    skewed and has kurtosis. Uses the Cornish-Fisher modification
    """
    # Compute the Z-score assuming it is Gaussian
    z=norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z=(z+
          (z**2-1)*s/6+
          (z**3 -3*z)*(k-3)/24-
          (2*z**3-5*z)*(s**2)/36)
    return -(r.mean()+z*r.std())


def cvar_historic(r, level=5):
    if isinstance(r, pd.Series):
        is_beyond=r<=-var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
        
def get_ind_returns(num, ew=False):
    """
    Load and format the Ken French 30 Industry Portfolios
    """
    s=str(num)
    if ew:
        st="ind"+str(num)+"_m_ew_rets.csv"
    else:
        st="ind"+str(num)+"_m_vw_rets.csv"
        
    ind=pd.read_csv(st,
               header=0, index_col=0,
               parse_dates=True)/100
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind


def get_ind_size():
    ind=pd.read_csv("ind30_m_size.csv",
               header=0, index_col=0,
               parse_dates=True)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind

def get_ind_market_caps(num, weights=False):
    s=str(num)
    indS=pd.read_csv("ind"+s+"_m_size.csv",
                    header=0, index_col=0,
                    parse_dates=True)
    indS.index=pd.to_datetime(indS.index, format="%Y%m").to_period('M')
    indS.columns=indS.columns.str.strip()
    
    indN=pd.read_csv("ind"+s+"_m_nfirms.csv",
                    header=0, index_col=0,
                    parse_dates=True)
    indN.index=pd.to_datetime(indN.index, format="%Y%m").to_period('M')
    indN.columns=indN.columns.str.strip()
    ind_caps=indS*indN
    if weights:
        tot_caps=ind_caps.sum(axis=1)
        ind_caps=ind_caps.divide(tot_caps, axis=0)
        
    return ind_caps
    

def get_ind_nfirms():
    
    ind=pd.read_csv("ind30_m_nfirms.csv",
               header=0, index_col=0,
               parse_dates=True)
    ind.index=pd.to_datetime(ind.index,format="%Y%m").to_period('M')
    ind.columns=ind.columns.str.strip()
    return ind

def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**.5)

def annualize_rets(r, periods_per_year):
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def compound(r):
    r=(1+r).prod()-1
    return r

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret, periods_per_year)
    ann_vol=annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def sortino_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period=(1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret=r-rf_per_period
    ann_ex_ret=annualize_rets(excess_ret, periods_per_year)
    ann_neg_vol=annualize_vol(r[r<0], periods_per_year)
    return ann_ex_ret/ann_neg_vol

    

def portfolio_return(weights, returns):
    """
    Weights vector ----> Returns vector
    VECTOR = SERIES
    """
    return weights.T @ returns

def portfolio_return2(weights, returns):
    """
    Weights vector ----> Returns vector
    VECTOR = SERIES
    """
    print(f'weights are {weights}\n')
    combined_rets=pd.DataFrame(returns@weights)
    monte_carlo_rets=np.random.normal(combined_rets.mean(), combined_rets.std(), (252*5, 500))
    total_cum_rets=(1+pd.DataFrame(monte_carlo_rets)).cumprod()
    final_rets=total_cum_rets.iloc[-1,-1]
    return -(final_rets)
    
def portfolio_return3(weights, returns, monte_carlo_rets):
    """
    Weights vector ----> Returns vector
    VECTOR = SERIES
    """
    #print(f'weights are {weights} and rets are {returns}\n')
    #combined_rets=pd.DataFrame(returns@weights)
    #monte_carlo_rets=np.random.normal(combined_rets.mean(), combined_rets.std(), (252*5, 500))
    total_cum_rets=(1+pd.DataFrame(monte_carlo_rets)).cumprod()
    final_rets=total_cum_rets.iloc[-1]
    lower,upper=norm.interval(.9, loc=final_rets.mean(), scale=final_rets.std())
    
    return lower

def portfolio_vol(weights, covmat):
    """
    weights vector ----> vol vector
    """
    return (weights.T @ covmat @ weights)**.5


def plot_ef2(n_points, er, cov, style=".-"):
    if er.shape[0]!=2:
        raise ValueError("Plot_Ef2 can only plot 2-asset frontiers")
    weights=[np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets= [portfolio_return(w, er) for w in weights]
    vols= [portfolio_vol(w, cov) for w in weights]
    ef=pd.DataFrame({"Returns": rets, "Vol": vols})
    return ef.plot.line(x="Vol", y="Returns", style=style)


def find_Beta(rets):
    corr_mat=rets.corr()
    location=rets.columns.get_loc('S&P 500')
    vols=rets.std()
    covmat=(corr_mat.mul(vols, axis=0)).mul(vols.T, axis=1)
    cov_market=pd.DataFrame(covmat.iloc[location,:])
    market_var=cov_market.iloc[location]
    beta_mat=cov_market/market_var[0]
    return beta_mat['S&P 500']
    
    
def find_Beta_auto(rets):
    corr_mat=rets.corr()
    vols=rets.std()
    covmat=rets.cov()
    cov_market=pd.DataFrame(covmat.iloc[-1,:])
    market_var=cov_market.iloc[-1]
    beta_mat=cov_market/market_var[0]
    return beta_mat['S&P 500']

def ride_winner(df, weight_for_winner):
    symbol=''
    val=0
    location=-1
    n=len(df.columns)
    rows=df.shape[0]
    rest=1-weight_for_winner
    weight_list=None
    weighted_rets=pd.DataFrame().reindex_like(df)
    weighted_rets=df.copy()
    for x in range(1, rows-1):
        val=(df.iloc[x]).max()
        symbol=(df.iloc[x]).idxmax()
        location=df.columns.get_loc(symbol)
        weight_list=pd.Series(rest/(n-1), range(0,n))
        weight_list.at[location]=weight_for_winner
        
        weighted_rets.iloc[x+1]=df.iloc[x+1]*weight_list.values
    return weighted_rets.sum(axis=1)

def minimize_vol(target_return, er, cov):
    """
    target_return -----> Weight vector
    """
    n=er.shape[0]
    initial_guess=np.repeat(1/n, n)
    #constraints
    bounds=((0,1), )*n
    return_is_target={
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights1, er: target_return-portfolio_return(weights1, er)
    }
    weights_sum_to_1={
        'type': 'eq',
        'fun': lambda weights1: np.sum(weights1)-1
    }
    weights=minimize(portfolio_vol, initial_guess,
                                  args=(cov,), method="SLSQP",
                                   options={'disp': False},
                                   constraints=(return_is_target, weights_sum_to_1),
                                   bounds=bounds
                                  )
    return weights.x


def optimal_weights(n_points, er, cov):
    """
    
    """
    target_returns=np.linspace(er.min(), er.max(), n_points)
    weights=[minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights

    
def plot_ef(n_points, er, cov, style=".-", show_cml=False, riskfree_rate=0, show_EWP=False,
           show_GMV=False):
    """
    Weights to minimize volatility for a given target return.
    """
    weights=optimal_weights(n_points, er, cov)
    rets= [portfolio_return(w, er) for w in weights]
    vols= [portfolio_vol(w, cov) for w in weights]
    ef=pd.DataFrame({
        "Returns": rets,
        "Vol": vols})
    ax=ef.plot.line(x="Vol", y="Returns", style=style)
    if show_EWP:
        n=er.shape[0]
        weights_EW=np.repeat(1/n, n)
        returns_EW=portfolio_return(weights_EW, er)
        vol_EW=portfolio_vol(weights_EW, cov)
        ax.plot([vol_EW], [returns_EW], color="goldenrod", marker="o", markersize=12)
 
    if show_GMV:
        n=er.shape[0]
        #GMV Weights only depend on covar matrix
        weights_GMV=gmv(cov)
        returns_GMV=portfolio_return(weights_GMV, er)
        vol_GMV=portfolio_vol(weights_GMV, cov)
        ax.plot([vol_GMV], [returns_GMV], color="midnightblue", marker="o", markersize=12)
 
        
    if show_cml:
        ax.set_xlim(left = 0)
        weights_msr=msr(riskfree_rate, er, cov)
        returns_msr=portfolio_return(weights_msr, er)
        vol_msr=portfolio_vol(weights_msr, cov)
        # Add CML
        cml_x_coord=[0, vol_msr]
        cml_y_coord=[riskfree_rate, returns_msr]
        ax.plot(cml_x_coord, cml_y_coord, color="green", marker="o", linestyle="dashed",
               markersize=12, linewidth=2)
    return ax
        

def msr(riskfree_rate, er, cov):
    """
    Risk Free Rate + ER + Cov -----> Weight vector
    """
    n=er.shape[0]
    initial_guess=np.repeat(1/n, n)
    #constraints
    bounds=((0,1), )*n
 
    weights_sum_to_1={
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        rets=portfolio_return(weights, er)
        vol=portfolio_vol(weights, cov)
        return -(rets-riskfree_rate)/vol

    results=minimize(neg_sharpe_ratio, initial_guess,
                                   args=(riskfree_rate, er, cov,), method="SLSQP",
                                   options={'disp': False},
                                   constraints=(weights_sum_to_1),
                                   bounds=bounds
                                  )
    return results.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Variance portfolio
    """
    n=cov.shape[0]
    return msr(0, np.repeat(1,n), cov)


def get_total_market_index_returns():
    ind_return=get_ind_returns()
    ind_nfirms=get_ind_nfirms()
    ind_size=get_ind_size()
    ind_mktcap=ind_nfirms*ind_size
    total_mktcap=ind_mktcap.sum(axis=1)
    ind_capweight=ind_mktcap.divide(total_mktcap, axis=0)
    total_market_return=(ind_capweight*ind_return).sum(axis=1)
    return total_market_return


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=.8,
            riskfree_rate=.03, drawdown=None):
    dates=risky_r.index
    n_steps=len(dates)
    account_value=start
    floor_value=start*floor
    peak=start
    
    if isinstance(risky_r, pd.Series):
        risky_r=pd.DataFrame(risky_r, columns=["R"])
        
    if safe_r is None:
        safe_r=pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:]=riskfree_rate/12
        
    account_history=pd.DataFrame().reindex_like(risky_r)
    cushion_history=pd.DataFrame().reindex_like(risky_r)
    risky_w_history=pd.DataFrame().reindex_like(risky_r)
    risky_alloc_history=pd.DataFrame().reindex_like(risky_r)

    if risky_r.loc[0,0]<1:
        risky_r=risky_r+1
        
    for step in range(n_steps):
        if drawdown is not None:
            peak=np.maximum(peak, account_value)
            floor_value=peak*(1-drawdown)
        cushion=(account_value-floor_value)/account_value
        risky_weight=m*cushion
        risky_weight=np.minimum(risky_weight, 1)
        risky_weight=np.maximum(risky_weight, 0)
        safe_weight=1-risky_weight
        risky_alloc=account_value*risky_weight
        safe_alloc=account_value*safe_weight
        ## Now update the account value for this single step
        
           
        account_value=risky_alloc*(risky_r.iloc[step])+safe_alloc*(1+safe_r.iloc[step])
        #Now save the values so we can look at history and plot it etc
        cushion_history.iloc[step]=cushion
        risky_w_history.iloc[step]=risky_weight
        account_history.iloc[step]=account_value
        risky_alloc_history.iloc[step]=risky_alloc
        
    risky_wealth=start*(risky_r).cumprod()
    backtest_result={
            "Wealth":account_history,
            "Risky Wealth": risky_wealth,
            "Risk Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "floor": floor,
            "risky_r": risky_r,
            "safe_r": safe_r,
            "risky_hist": risky_alloc_history
        }
    return backtest_result
    
    
def summary_stats(r, riskfree_rate=.03):
    ann_r=r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol=r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr=r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd=r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew=skewness(r)
    kurt=kurtosis(r)
    cf_var5=var_gaussian(r)
    hist_cvar5=cvar_historic(r)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })



def gbm1(n_years=10, n_scenarios=1000, mu=.07, sigma=.15, steps_per_year=12, s_0=100, prices=False):
    """
    Evolution of stock price using Geometric Brownian Motion Model
    """
    dt=1/steps_per_year
    n_steps=int(n_years*steps_per_year)+1
    rets_plus1=np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    rets_plus1[0]=1
    # Rets to Prices
    if not prices:
        prices=pd.DataFrame(rets_plus1)
    else:
        prices=s_0*pd.DataFrame(rets_plus1)
    return prices


def show_cppi(n_scenarios=50, mu=.07, sigma=.15, m=3, floor=0, riskfree_rate=.03, y_max=100, x_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start=100
    sim_rets=gbm1(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
    risky_r=sim_rets.copy()
    ## run the back test
    btr=run_cppi(risky_r=risky_r, riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth=btr['Wealth']
    print(wealth.iloc[0,0])
    y_max=wealth.values.max()*y_max/100
    print(y_max)
    terminal_wealth=wealth.iloc[-1]
    tw_mean=terminal_wealth.mean()
    tw_median=terminal_wealth.median()
    failure_mask=np.less(terminal_wealth, start=floor)
    n_failures=failure_mask.sum()
    p_fail=n_failures/n_scenarios
    exp_shortfall=np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures>0 else 0
    
    # PLOT
    fig, (wealth_ax, hist_ax)=plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}
                                          ,figsize=(24,9))
    plt.subplots_adjust(wspace=0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=.1, color="indianred")
    
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls=":", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc="indianred",
                              orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")


def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time=t
    r is per period. returns a t * r series or DF. r can be float, series, or DF.
    returns DF indexed by t
    """
    discounts=pd.DataFrame([(r+1)**-i for i in t])
    discounts.index=t
    return discounts


def pv(flows, r):
    """
    Computes PV of sequence of Liabilities. l is indexed by the time, and the values are the amounts of each Lia. returns PV
    """
    dates=flows.index
    discounts=discount(dates, r)
    return discounts.multiply(flows, axis=0).sum()


def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio
    """
    return pv(assets, r)/pv(liabilities, r)


def show_funding_ratio(liabilities, assets, r):
    fr=funding_ratio(assets, liabilities, r)
    print(f'{fr*100:.2f}')
    
    
def inst_to_ann(r):
    """
    Converts short rate to annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert annualized to short rate
    """
    return np.log1p(r)


def cir1(n_years=10, n_scenarios=1, a=.05, b=.03, sigma=.05, steps_per_year=12, r_0=None):
    """
    Generate Random int rate evolution over time using the CIR model
    b and r_ are assumed to be annualized rates, not short rates
    returns annualized rates also
    """
    if r_0 is None:
        r_0=b
        
    r_0=ann_to_inst(r_0)
    dt=1/steps_per_year
    num_steps=int(n_years*steps_per_year)+1
    
    shock=np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates=np.empty_like(shock)
    rates[0]=r_0
    
    h=math.sqrt(a**2 + 2*sigma**2)
    prices=np.empty_like(shock)
    
    def price(ttm, r):
        _A=((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B=(2*(math.exp(h*ttm)-1))/(2*h+(h+a)*(math.exp(h*ttm)-1))
        _P=_A*np.exp(-_B*r)
        return _P
    prices[0]=price(n_years, r_0)
    
    
    for step in range(1, num_steps):
        r_t=rates[step-1]
        d_r_t=a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step]=abs(r_t+d_r_t)
        #Generate prices at time t as well...
        prices[step]=price(n_years-step*dt, rates[step])
        

    rates= pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices=pd.DataFrame(data=prices, index=range(num_steps))
    return rates, prices

def bond_cash_flows(maturity, principal=100, coupon_rate=.03, coupons_per_year=12):
    """
    Returns series of cash flows from bond. Indexed by coupon number
    """
    n_coupons=round(maturity*coupons_per_year)
    coupon_amt=principal*coupon_rate/coupons_per_year
    coupon_times=np.arange(1, n_coupons+1)
    cash_flows=pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1]+=principal
    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=.03, coupons_per_year=12, discount_rate=.03):
    """
    Price bond based on parameters. uses DataFrame with rates, and produces dataframe of bond
    prices. so accounts for many rate scenarios
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates=discount_rate.index
        prices=pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t]=bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else:  #Base case= 1 time period
        if maturity<=0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows=bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

    
def macaulay_duration(flows, discount_rate):
    discounted_flows=discount(flows.index, discount_rate)*flows
    weights=discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)
    
    

def MacDur(cash_flows, discount_rate):
    discounted_flows=discount(cash_flows.index, discount_rate)*pd.DataFrame(data=cash_flows, index=cash_flows.index)
    weights=discounted_flows/discounted_flows.sum()
    return np.average(cash_flows.index, weights=weights[0])

def match_durations(cf_t, cf_s, cf_l, discount_rate, *args):
    """
    finds weights of bonds to equal Lia MacDur. Returns weight in Shorter Dur Bond
    """
    for a in args:
        print(a)
    d_t=MacDur(cf_t, discount_rate)   # Liability
    d_s=MacDur(cf_s, discount_rate/args[1])/args[1] #short bond
    d_l=MacDur(cf_l, discount_rate/args[0])/args[0]  #Long Bond
    return (d_l -d_t)/(d_l-d_s)


def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    TR of bond. Assumes divs are paid at end of period, and dividends are reinvested
    """
    coupons=pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max=monthly_prices.index.max()
    pay_date=np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date]=principal*coupon_rate/coupons_per_year
    total_returns=(monthly_prices+coupons)/monthly_prices.shift()-1
    return total_returns.dropna()



def fixed_mix_allocator(r1, r2, w1, **kwargs):
    """
    PSP and LHP are T x N DataFrames. EAch column=scenario. row=prices
    Returns T x N DataFrame of PSP weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)


def bt_mix(rets1, rets2, allocator, **kwargs):
    """
    Runs a back test by allocating betweeen 2 sets of returns. r1 and r2 are DataFrames T x N. T=time
    N=scenarios
    Allocator takes 2 sets of returns and other args, and spits out an allocation (weights)
    Allocator returns a T x 1 DataFrame with weights. then bt_mix returns T x N DataFrame
    """
    if not rets1.shape==rets2.shape:
        raise ValueError("r1 and r2 must be same shape")
    
    weights=allocator(rets1, rets2, **kwargs)
    if not weights.shape==rets1.shape:
        raise ValueError("Allocator returned wrong shape")
    rets_mix=weights*rets1+(1-weights)*rets2
    return rets_mix


def terminal_values(rets):
    """
    returns final values of $1.00 at end of return period for each scenario
    """
    return (1+rets).prod()


def terminal_stats(rets, floor=.8, cap=np.inf, name="Stats"):
    """
    Summary stats on TV per dollar across N scenarios
    rets=T x N DF of returns. Returns 1 column DF, indexed by Stat name
    """
    terminal_wealth=(1+rets).prod()
    breach=terminal_wealth<floor
    reach=terminal_wealth>=cap
    p_breach=breach.mean() if breach.sum()>0 else np.nan
    p_reach=reach.mean() if reach.sum()>0 else np.nan
    e_short=(floor-terminal_wealth[breach]).mean() if breach.sum()>0 else np.nan
    e_surplus=(cap-terminal_wealth[reach]).mean() if reach.sum()>0 else np.nan
    sum_stats=pd.DataFrame.from_dict({
        'mean': terminal_wealth.mean(),
        'std': terminal_wealth.std(),
        'p_breach': p_breach,
        'e_short': e_short,
        'p_reach': p_reach,
        'e_surplus': e_surplus
    }, orient='index', columns=[name])
    return sum_stats


def glide_path_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates Target Date Fund gradual move from r1 to r2
    """
    
    n_points=r1.shape[0]
    n_col=r1.shape[1]
    path=pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths=pd.concat([path]*n_col, axis=1)
    paths.index=r1.index
    paths.columns=r1.columns
    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP, and don't violate floor. Uses CPPI style dynamic risk
    budgeting, by investing in multiple of the cushion in the PSP.
    """
    if zc_prices.shape !=psp_r.shape:
        raise ValueError("PSP and ZC Prices must have same shape")
    n_steps, n_scenarios=psp_r.shape
    account_value=np.repeat(1, n_scenarios)
    floor_value=np.repeat(1, n_scenarios)
    w_history=pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value=floor*zc_prices.iloc[step] # PV of Floor assuming today's rates and flat YC
        cushion=(account_value-floor_value)/account_value
        psp_w=(m*cushion).clip(0,1) # same as applying min and max
        ghp_w=1-psp_w
        psp_alloc=account_value*psp_w
        ghp_alloc=account_value*ghp_w
        # Now recompute new account value
        account_value=psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step]=psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP, and don't violate floor. Uses CPPI style dynamic risk
    budgeting, by investing in multiple of the cushion in the PSP.
    """
    n_steps, n_scenarios=psp_r.shape
    account_value=np.repeat(1, n_scenarios)
    floor_value=np.repeat(1, n_scenarios)
    peak_value=np.repeat(1, n_scenarios)
    w_history=pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value=(1-maxdd)*peak_value # PV of Floor assuming today's rates and flat YC
        cushion=(account_value-floor_value)/account_value
        psp_w=(m*cushion).clip(0,1) # same as applying min and max
        ghp_w=1-psp_w
        psp_alloc=account_value*psp_w
        ghp_alloc=account_value*ghp_w
        # Now recompute new account value
        account_value=psp_alloc*(1+psp_r.iloc[step])+ghp_alloc*(1+ghp_r.iloc[step])
        peak_value=np.maximum(peak_value, account_value)
        w_history.iloc[step]=psp_w
    return w_history


def regress(dep_var, exp_var, alpha=True):
    """
    Runs linear regression
    Returns a statsmodel's RegressionResults object, and we can call .summary, .params, .t and .pvalues, .rsquared
    .rsquared_adj
    """
    if alpha:
        exp_var=exp_var.copy()
        exp_var['Alpha'==1]
        
    lm=sm.OLS(dep_var, exp_var).fit()
    return lm


def style_analysis(dep_var, exp_var):
    """
    Returns the optimal weights that minimize Tracking error between portfolio of explanatory vars and dep_var
    """
    n=exp_var.shape[1]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    # Constraints
  #  weights_value={'type': 'ineq',
   #               'fun': lambda weights: weights-0
    #              }
    weights_sum_to_1={'type': 'eq',
                     'fun': lambda weights: np.sum(weights)-1
                     }
    solution=minimize(portfolio_tracking_error, init_guess,
                      args=(dep_var, exp_var,),method='SLSQP',
                      options={'disp': False},
                      constraints=(weights_sum_to_1,),
                      bounds=bounds)
    
    weights=pd.Series(solution.x, index=exp_var.columns)
    return weights


def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns TE between reference returns and portfolio of building block returns held with given weights
    """
    bb_r_tot=bb_r
    N=ref_r.shape[0]
    if not weights.any()==0:
        bb_r_tot=(weights*bb_r).sum(axis=1)
    rets_diff=((ref_r-bb_r_tot)**2).sum()
    
    tr_er_1=rets_diff
    tr_er=np.sqrt(tr_er_1)
    return tr_er


def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())


def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on rets as a DataFrame
    """
    n=len(r.columns)
    ew=pd.Series(1/n, index=r.columns)
    
    if cap_weights is not None:
        cw=cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude Microcaps
        if microcap_threshold is not None and microcap_threshold>0:
            microcap=cw<microcap_threshold
            ew[microcap]=0
            ew=ew/ew.sum()
        #Limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult>0:
            ew=np.minimum(ew, cw*max_cw_mult)
            ew=ew/ew.sum()
    return ew
        
def getMSRWeights(r, **kwargs):
    weights=msr(0, r.mean(), r.cov())
    return pd.Series(weights, index=r.columns)


    
    
def weight_ema(rets, n, alpha=.2):
    if n==0:
        return rets.iloc[n]
    return alpha*rets.iloc[n]+(1-alpha)*weight_ema(rets, n-1, alpha)
    
    
def weight_ema3(rets, n, k=2/22):
    
    if n==0:
        return rets.iloc[n]
    return k*rets.iloc[n]+(1-k)*weight_ema3(rets, n-1)
    

def weight_cw(r, cap_weights, **kwargs):
    """
    just returns DF of cap weights at the first point in time that we have returns (r)
    """
    w=cap_weights.loc[r.index[0]]
    return w/w.sum()

def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters
    estimation window= months.  weighting=weighting scheme to use. must take rets and a variable # of kwargs
    """
    n_periods=r.shape[0]
    windows=[(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    weights=pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)
    #print(weights)
    returns=(weights*r).sum(axis=1, min_count=1)
    return weights,returns
    

def gmv_Helper(r, **kwargs):
    cov=r.cov()
    n=cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

    
def sample_cov(r, **kwargs):
    """
    Returns sample covariance of returns
    """
    return r.cov()


def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces weights of GMV portfolio given a covariance matrix of the returns and a cov estimator method
    """
    est_cov=cov_estimator(r, **kwargs)
    return gmv(est_cov)



#import statsmodels.stats.moment_helpers as mh
def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    ccov = ccor * np.outer(sd, sd)
#     mh.corr2cov(ccor, sd)
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)


def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample


def as_colvec(x):
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)
    
    
def implied_returns(delta, sigma, w):
    """
Obtain the implied expected returns by reverse engineering the weights
Inputs:
delta: Risk Aversion Coefficient (scalar)
sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
Returns an N x 1 vector of Returns as Series
    """
    #ir=pd.Series(delta*np.dot(sigma, w).squeeze())
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir



# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    #one=np.dot(p, tau*sigma)
    #tot=np.dot(one, p.T)
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    #tot=pd.DataFrame(tot)
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)





from numpy.linalg import inv

def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
# Computes the posterior expected returns based on 
# the original black litterman reference model
#
# W.prior must be an N x 1 vector of weights, a Series
# Sigma.prior is an N x N covariance matrix, a DataFrame
# P must be a K x N matrix linking Q and the Assets, a DataFrame
# Q must be an K x 1 vector of views, a Series
# Omega must be a K x K matrix a DataFrame, or None
# if Omega is None, we assume it is
#    proportional to variance of the prior
# delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)



# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w


def w_star(delta, sigma, mu):
    return (inverse(sigma).dot(mu))/delta


def risk_contribution(w,cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib



def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)

def equal_risk_contributions_Helper(r):
    return equal_risk_contributions(r.cov())