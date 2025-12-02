import os
import pickle
import warnings
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from processing import get_features_targets_from_data, prepare_features

# Ignore future warnings.
warnings.simplefilter(action='ignore', category=(FutureWarning))


def boxplot_predictions(y_predict, y_test, size=9, figsize=(6.5,3.8), 
                        title='', set_title=False, savefig=False):
    """
    Plot model predictions' statistical distributions.

    Bayesian model predictions return statistical probability
    distributions, instead of point values, which are then
    visualized using a boxplot against true values.

    Parameters
    ----------
    y_predict : 2d-array
        Model predictions. This is a 2d array where its size 
        is defined by the number of samples from the MCMC chain 
        and the number of cells (targets) in the test set.
    y_test : 1d-array
        EoL cell values (targets) from the test set.
    size : int, default=9
        Number of cells from the test set for which the statistical
        probability distributions of predictions will be shown on
        the plot. This is for reducing the visual clutter where
        test set has many instances.
    figsize : tuple[real, real], default=(6.5, 3.8)
        Dimensions of the figure (in inches, unfortunately).
    title : str, default=''
        Title of the figure and a name of the file when the figure
        is saved (as a 600 dpi .png file).
    set_title : bool, default=False
        Indicator for setting the title in the figure.
    savefig : bool, default=False
        Indicator for saving the figure. File name will be the
        string from the `title` parameter. Figure is saved as
        a .png file with a 600 dpi resolution.
    
    Returns
    -------
        Matplotlib figure. If the option of saving the figure
        is selected, then the figure will be also saved as a
        .png file on disk with a 600 dpi resolution.
    """
    # Random subset of results from the test set.
    rng = np.random.default_rng(seed=356)
    ids = rng.choice(len(y_test), size=size, replace=False, shuffle=False)
    # Determine y-axis max. limit.
    q = np.quantile(y_predict[:,ids], q=0.9, axis=0)
    maxval = max(y_test[ids].max() + 100, 1.5*q.max())
    limit = min(maxval, 2000)
    support = np.arange(size)
    # Plot predictions from the test data subset.
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(which='major', axis='y', color='lightgrey', alpha=0.5)
    if set_title:
        ax.set(axisbelow=True, title=title)
    ax.boxplot(y_predict[:,ids], positions=support, 
               widths=0.75, showmeans=False, showfliers=True,
               flierprops={'marker': '+', 'markersize': 5, 
                           'markeredgecolor': 'lightslategrey'},
               medianprops={'color': 'darkslategrey', 'linewidth': 1.5}, 
               label='predictions')
    ax.scatter(support, y_test[ids], marker='d', s=40, c='navy', 
               label='true values', zorder=99)
    ax.axhline(550, linewidth=1, linestyle='--', color='darkorange')
    ax.legend(loc='upper right')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('EoL (cycles)')
    ax.set_ylim(0, limit)
    fig.tight_layout()
    if savefig:
        # Save figure on disk.
        plt.savefig(title+'.png', dpi=600)
    plt.show()


# ----------------------------------------------------------------------------
# Input parameters:
# ----------------------------------------------------------------------------
targets = 'eol'  # ['eol', 'knee', 'knee-onset']
set_of_features = 'discharge'  # ['variance', 'discharge', 'full', 'custom', 'all', 'vif']

print('Importing data ...')
# Import data from the external file.
path = os.path.dirname('/home/ps/Documents/Batteries/Data/')  # EDIT HERE
# Data is stored as a pickle file.
features_file = 'batchdata_all_features.pkl'  # EDIT HERE
with open(file=os.path.join(path, features_file), mode='rb') as fp:
    X_data_dict = pickle.load(fp)
targets_file = 'batchdata_all_targets.pkl'  # EDIT HERE
with open(file=os.path.join(path, targets_file), mode='rb') as tp:
    y_data_dict = pickle.load(tp)

# Prepare features.
X_data = prepare_features(X_data_dict, set_of_features)
selected_features = X_data.columns
# Prepare targets.
y_data = y_data_dict[targets]

# Split dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, train_size=0.8, shuffle=True,
    #stratify=np.where(y_data < 550, 0 ,1)
)
# Standardize data sets.
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

# ----------------------------------------------------------------------------
# Univariate multilevel robust linear regression.
# ----------------------------------------------------------------------------
print('\nUnivariate multilevel robust linear model:')
idx_train = np.where(y_train < 550, 0, 1)
idx_test = np.where(y_test < 550, 0, 1)
# Partial pooling model.
with pm.Model() as univar:
    # Data containers.
    x_data = pm.Data('x_data', X_train['std'].values)
    y_data = pm.Data('y', y_train)
    idx = pm.Data('idx', idx_train)
    # Hyperpriors.
    ma = pm.Normal('ma', mu=0, sigma=10)
    sa = pm.HalfCauchy('sa', beta=1)
    mb = pm.Normal('mb', mu=0, sigma=10)
    sb = pm.HalfCauchy('sb', beta=1)
    # Priors.
    alpha = pm.Normal('alpha', mu=ma, sigma=sa, shape=2)
    beta = pm.Normal('beta', mu=mb, sigma=sb, shape=2)
    sigma = pm.HalfCauchy('sigma', beta=1)
    nu = pm.Exponential('nu', lam=1)
    # Linear model.
    mu = alpha[idx] + x_data * beta[idx]
    # Likelihood.
    obs = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, 
                      observed=np.log10(y_data), shape=y_data.shape)
    # MCMC sampling.
    idata_uni = pm.sample(draws=4000, tune=2000, target_accept=0.99,
                          idata_kwargs={'log_likelihood': True})

print(az.summary(idata_uni))

# Alternative formulation (with a multivariate prior).
with pm.Model() as mvn:
    # Data containers.
    x_data = pm.Data('x_data', X_train['std'].values)
    y_data = pm.Data('y', y_train)
    idx = pm.Data('idx', idx_train)
    # Multivariate normal prior for the intercepts and slopes.
    sd_dist = pm.HalfCauchy.dist(beta=2)
    chol, corr, stds = pm.LKJCholeskyCov('chol', eta=2, n=2, sd_dist=sd_dist)
    ab = pm.MvNormal('ab', mu=0, chol=chol, shape=(2,2))
    alpha = pm.Deterministic('alpha', ab[:,0])  # intercept
    beta = pm.Deterministic('beta', ab[:,1])    # slope
    sigma = pm.HalfCauchy('sigma', beta=1)
    nu = pm.Exponential('nu', lam=1)
    # Linear model.
    mu = alpha[idx] + x_data * beta[idx]
    # Likelihood.
    obs = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, 
                      observed=np.log10(y_data), shape=y_data.shape)
    # MCMC sampling.
    idata_mvn = pm.sample(draws=4000, tune=2000, target_accept=0.99,
                          idata_kwargs={'log_likelihood': True})

print(az.summary(idata_mvn, var_names=['nu', 'alpha', 'beta', 'sigma']))

with univar:
    # Posterior predictive samples for model checking.
    idata_uni.extend(pm.sample_posterior_predictive(idata_uni))

fig, ax = plt.subplots(figsize=(5, 3))
az.plot_ppc(idata_uni, num_pp_samples=500, textsize=11, ax=ax)
ax.set_xlim(2, 3.5)
fig.tight_layout()
plt.savefig('ppc_uni.png', dpi=600)
plt.show()

# Compare posterior predictions with actual values
idx = idx_train
alpha = idata_uni.posterior['alpha'].mean(dim=('chain')).values
beta = idata_uni.posterior['beta'].mean(dim=('chain')).values
support = np.linspace(X_train['std'].values.min(), X_train['std'].values.max(), 100)
y0 = 10**(alpha[:,0] + support.reshape(-1, 1) * beta[:,0])
y1 = 10**(alpha[:,1] + support.reshape(-1, 1) * beta[:,1])

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.set_yscale('log')
ax.scatter(X_train['std'].values[idx==0], y_train[idx==0], marker='o', s=20, 
           c='orange', edgecolor='dimgrey', alpha=0.75, label='short life')
ax.plot(support, y0.mean(1), ls='-', lw=1.5, c='darkorange', 
        label='linear fit (short)')
az.plot_hdi(support, y0.T, hdi_prob=0.97, color='wheat', ax=ax)
ax.scatter(X_train['std'].values[idx==1], y_train[idx==1], marker='o', s=20, 
           c='royalblue', edgecolor='dimgrey', alpha=0.75, label='long life')
ax.plot(support, y1.mean(1), ls='-', lw=1.5, c='darkblue', 
        label='linear fit (long)')
az.plot_hdi(support, y1.T, hdi_prob=0.97, color='lightblue', ax=ax)
ax.axhline(550, linewidth=1, linestyle='--', color='dimgrey')
ax.text(support.min(), 560, u'\u21A5'+'long life', fontsize=10, color='royalblue')
ax.text(support.min(), 490, u'\u21A7'+'short life', fontsize=10, color='darkorange')
ax.legend(loc='upper right')
ax.grid(which='both', axis='both', linewidth=0.5, color='lightgrey')
ax.set_xlabel(r'$\log_{10}(Var(\Delta Q_{100-10}))$')
ax.set_ylabel('EoL (cycles)')
ax.set_ylim(bottom=100)
fig.tight_layout()
plt.savefig('Univariate regression full.png', dpi=600)
plt.show()

with univar:
    pm.set_data({'x_data': X_test['std'].values})
    pm.set_data({'y': y_test})
    pm.set_data({'idx': idx_test})
    y_predict = pm.sample_posterior_predictive(idata_uni, predictions=True)

# Predictions on the test set.
y_predict_values = 10**y_predict.predictions['likelihood'].mean(dim='chain').values
boxplot_predictions(y_predict_values, y_test, title='univariate_regression',
                    savefig=True)

# Extract mean predicitons.
y_pred = y_predict.predictions['likelihood'].mean(dim=['chain', 'draw']).values
y_pred = 10**y_pred  # return back to targets

# RMSE and MAPE errors.
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE (test): {rmse:.2f} cycles')
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (test): {mape*100:.2f} %')

# ----------------------------------------------------------------------------
# Multiple ordinary linear regression model.
# ----------------------------------------------------------------------------
coords = {
    'samples': range(len(y_train)),
    'features': selected_features,
}
print('\nOrdinary linear regression:')
with pm.Model(coords=coords) as multi_linreg:
    # Data container.
    x_data = pm.Data('data', X_train, dims=('samples', 'features'))
    # Priors.
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, dims='features')
    sigma = pm.HalfCauchy('sigma', beta=2)
    # Linear model.
    mu = alpha + x_data @ betas
    # Likelihood.
    obs = pm.Normal('likelihood', mu=mu, sigma=sigma, 
                    observed=np.log10(y_train), dims='samples')
    # MCMC sampling.
    idata = pm.sample(2000, target_accept=0.95,
                      idata_kwargs={'log_likelihood': True})

print(az.summary(idata))

# Alternative formulation (with a multivariate prior on slopes).
with pm.Model() as mv:
    # Data containers.
    x_data = pm.Data('x_data', X_train.values)
    y_data = pm.Data('y', y_train)
    idx = pm.Data('idx', idx_train)
    # Multivariate normal prior for the intercepts and slopes.
    sd_dist = pm.HalfCauchy.dist(beta=2)
    chol, corr, stds = pm.LKJCholeskyCov('chol', eta=2, n=6, sd_dist=sd_dist)
    betas = pm.MvNormal('betas', mu=0, chol=chol, shape=6)
    alpha = pm.Normal('alpha', mu=0, sigma=10)  # intercept
    sigma = pm.HalfCauchy('sigma', beta=1)
    nu = pm.Exponential('nu', lam=1)
    # Linear model.
    mu = alpha + x_data @ betas
    # Likelihood.
    obs = pm.Normal('likelihood', mu=mu, sigma=sigma, 
                    observed=np.log10(y_train), shape=y_data.shape)
    # MCMC sampling.
    idata_mv = pm.sample(2000, target_accept=0.99, max_treedepth=15,
                         idata_kwargs={'log_likelihood': True})

print(az.summary(idata_mv))

with multi_linreg:
    # Posterior predictive samples for model checking.
    idata.extend(pm.sample_posterior_predictive(idata))

fig, ax = plt.subplots(figsize=(5, 3))
az.plot_ppc(idata, num_pp_samples=500, textsize=11, ax=ax)
fig.tight_layout()
plt.savefig('ppc_ols.png', dpi=600)
plt.show()

# Change the data container dimension for the test set.
coords_test = coords | {'samples': range(len(y_test))}
# Predict on test data.
with multi_linreg:
    # Import test data for predictions.
    pm.set_data({'data': X_test}, coords=coords_test)
    # Predict on test set data.
    y_predict = pm.sample_posterior_predictive(idata, predictions=True)
y_predict_values = 10**y_predict.predictions['likelihood'].mean(dim='chain').values
boxplot_predictions(y_predict_values, y_test, 
                    title='Ordinary linear regression (OLR)', savefig=True)
# Extract mean predicitons.
y_pred = y_predict.predictions['likelihood'].mean(dim=['chain', 'draw']).values
y_pred = 10**y_pred  # return back to targets

# RMSE and MAPE errors.
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE (test): {rmse:.2f} cycles')
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (test): {mape*100:.2f} %')

# ----------------------------------------------------------------------------
# Robust multiple linear regression model.
# ----------------------------------------------------------------------------
print('\nRobust linear regression:')
with pm.Model(coords=coords) as robust:
    # Data container.
    x_data = pm.Data('data', X_train, dims=('samples', 'features'))
    # Priors.
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, dims='features')
    sigma = pm.HalfStudentT('sigma', nu=4, sigma=1)
    # Linear model.
    mu = alpha + x_data @ betas
    # Likelihood.
    obs = pm.StudentT('likelihood', nu=2, mu=mu, sigma=sigma, 
                      observed=np.log10(y_train), dims='samples')
    # MCMC sampling.
    idata_robust = pm.sample(2000, target_accept=0.95,
                             idata_kwargs={'log_likelihood': True})

# Predict on test data.
with robust:
    # Import test data for predictions.
    pm.set_data({'data': X_test}, coords=coords_test)
    # Predict on test set data.
    y_predict = pm.sample_posterior_predictive(idata_robust, predictions=True)

y_predict_values = 10**y_predict.predictions['likelihood'].mean(dim='chain').values
boxplot_predictions(y_predict_values, y_test, 
                    title='Robust linear regression (RLR)', savefig=True)
# Extract mean predicitons.
y_pred = y_predict.predictions['likelihood'].mean(dim=['chain', 'draw']).values
y_pred = 10**y_pred  # return back to targets

# RMSE and MAPE errors.
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE (test): {rmse:.2f} cycles')
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (test): {mape*100:.2f} %')

# ----------------------------------------------------------------------------
# Generalized linear model (GLM): Gamma distribution with a log-link function.
# ----------------------------------------------------------------------------
print('\nGeneralized linear model:')
with pm.Model(coords=coords) as glm:
    # Data container.
    x_data = pm.Data('data', X_train, dims=('samples', 'features'))
    # Priors.
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, dims='features')
    sigma = pm.HalfCauchy('sigma', beta=1)
    # Linear model.
    mu = pm.math.exp(alpha + x_data @ betas)
    # Likelihood.
    obs = pm.Gamma('likelihood', mu=mu, sigma=sigma, 
                   observed=np.log10(y_train), dims='samples')
    # MCMC sampling.
    idata_glm = pm.sample(2000, target_accept=0.95,
                          idata_kwargs={'log_likelihood': True})

with glm:
    # Sample the posterior predictive distribution.
    idata_glm.extend(pm.sample_posterior_predictive(idata_glm))

fig, ax = plt.subplots(figsize=(5, 3))
az.plot_ppc(idata_glm, num_pp_samples=500, textsize=11, ax=ax)
fig.tight_layout()
plt.savefig('ppc_glm.png', dpi=600)
plt.show()

# Predict on test data.
with glm:
    # Import test data for predictions.
    pm.set_data({'data': X_test}, coords=coords_test)
    # Predict on test set data.
    y_predict = pm.sample_posterior_predictive(idata_glm, predictions=True)

y_predict_values = 10**y_predict.predictions['likelihood'].mean(dim='chain').values
boxplot_predictions(y_predict_values, y_test, 
                    title='Generalized linear model (GLM)', savefig=True)
# Extract mean predicitons.
y_pred = y_predict.predictions['likelihood'].mean(dim=['chain', 'draw']).values
y_pred = 10**y_pred  # return back to targets

# RMSE and MAPE errors.
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE (test): {rmse:.2f} cycles')
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (test): {mape*100:.2f} %')

# ----------------------------------------------------------------------------
# Multilevel generalized multiple linear regression model.
# ----------------------------------------------------------------------------
print('\nMultilevel generalized linear model:')
# Two clusters of cells with a threshold at 550 cycles.
# Index for "short" and "long" EoL values.
# 0 - "short", 1 - "long"
idx_train = np.where(y_train < 550, 0, 1)
idx_test = np.where(y_test < 550, 0, 1)
c = {
    'index': range(2),
    'samples': range(len(y_train)),
    'features': selected_features,
}
with pm.Model(coords=c) as multi_level_glm:
    # Data containers.
    x_data = pm.Data('data', X_train, dims=('samples', 'features'))
    y_data = pm.Data('y', y_train, dims='samples')
    idx = pm.Data('idx', idx_train, dims='samples')
    # Hyperpriors.
    m = pm.Normal('m', mu=0, sigma=10)
    s = pm.HalfCauchy('s', beta=1)
    sigma_j = pm.HalfCauchy('sigma_j', beta=1)
    # Priors.
    alpha = pm.Normal('alpha', mu=m, sigma=s, dims='index')
    beta = pm.StudentT('beta', nu=4, mu=0, sigma=sigma_j, dims='features')
    sigma = pm.HalfCauchy('sigma', beta=1)
    # Linear model with varying intercepts (log-link function).
    mu = pm.math.exp(alpha[idx] + x_data @ beta)
    # Likelihood.
    obs = pm.Gamma('likelihood', alpha=mu**2/sigma**2, beta=mu/sigma**2, 
                   observed=np.log10(y_data), dims='samples')
    # MCMC sampling.
    idata_multi_glm = pm.sample(draws=4000, tune=2000, 
                                target_accept=0.99, max_treedepth=15,
                                idata_kwargs={'log_likelihood': True})

# Alternative: Varying slopes model with a multivariate prior.
fts = ['std', 'min', 'skew', 'kurt']
with pm.Model(coords=c) as multi_level_glm:
    n = X_train.values.shape[1]
    # Data containers.
    x_data = pm.Data('data', X_train[fts], dims=('samples', 'features'))
    y_data = pm.Data('y', y_train, dims='samples')
    idx = pm.Data('idx', idx_train, dims='samples')
    # Hyperpriors on intercept.
    m = pm.Normal('m', mu=0, sigma=10)
    s = pm.HalfCauchy('s', beta=1)
    alpha = pm.Normal('alpha', mu=m, sigma=s, dims='index')
    # Multivariate normal prior for the slopes.
    sd_dist = pm.HalfCauchy.dist(beta=2)
    chol, corr, stds = pm.LKJCholeskyCov('chol', eta=2, n=n, sd_dist=sd_dist)
    beta = pm.MvNormal('betas', mu=0, chol=chol, dims=('index', 'features'))
    sigma = pm.HalfCauchy('sigma', beta=1)
    # Linear model with varying intercepts and slopes (log-link function).
    mu = pm.math.exp(
        alpha[idx] 
        + x_data[:,0] * beta[:,0][idx]
        + x_data[:,1] * beta[:,1][idx]
        + x_data[:,2] * beta[:,2][idx]
        + x_data[:,3] * beta[:,3][idx]
    )
    # Likelihood.
    obs = pm.Gamma('likelihood', alpha=mu**2/sigma**2, beta=mu/sigma**2, 
                   observed=np.log10(y_data), dims='samples')
    # MCMC sampling.
    idata_multi_glm = pm.sample(draws=2000, tune=2000, 
                                target_accept=0.99, max_treedepth=15,
                                idata_kwargs={'log_likelihood': True})

# Posterior predictive check.
with multi_level_glm:
    idata_multi_glm.extend(pm.sample_posterior_predictive(idata_multi_glm))

# Plot retrodictions.
boxplot_predictions(10**idata_multi_glm.posterior_predictive['likelihood']
                    .mean(dim='chain').values,
                    y_train, size=30, figsize=(9.5, 3.5), title='retrodictions')

az.plot_trace(idata_multi_glm, var_names=['alpha', 'beta', 'sigma'], 
              figsize=(10, 5), combined=True);

print(az.summary(idata_multi_glm))

az.plot_posterior(idata_multi_glm, var_names=['alpha'], 
                  #ref_val=0, ref_val_color='slategrey',
                  textsize=11, figsize=(6, 2.8), lw=2);

fig, ax = plt.subplots(figsize=(5, 3))
az.plot_ppc(idata_multi_glm, num_pp_samples=500, textsize=11, ax=ax)
fig.tight_layout()
plt.savefig('ppc_multi_glm.png', dpi=600)
plt.show()

c_test = c | {'samples': range(len(y_test))}
# Predict on test data.
with multi_level_glm:
    # Import test data for predictions.
    pm.set_data({'data': X_test}, coords=c_test)
    pm.set_data({'y': y_test}, coords=c_test)
    pm.set_data({'idx': idx_test}, coords=c_test)
    # Predict on test set data.
    y_predict = pm.sample_posterior_predictive(idata_multi_glm, predictions=True)

y_predict_values = 10**y_predict.predictions['likelihood'].mean(dim='chain').values
boxplot_predictions(y_predict_values, y_test,
                    title='Multilevel GLM regression (MGLM)', savefig=True)
# Extract mean predicitons.
y_pred = y_predict.predictions['likelihood'].mean(dim=['chain', 'draw']).values
y_pred = 10**y_pred  # return back to targets

# Plot observed against predicted cycle lives,
# along with model residuals.
pred_interval = az.hdi(y_predict_values, hdi_prob=0.89)
fig, ax = plt.subplots(figsize=(4, 3.8))
ax.grid(which='major', axis='both', color='lightgrey', alpha=0.5)
ax.plot([0, 1], [0, 1], ls='--', lw=1.5, c='dimgrey',  # diagonal line
        transform=ax.transAxes)
ax.vlines(y_test, ymin=pred_interval[:,0], ymax=pred_interval[:,1],
          colors='royalblue', linestyles='solid', label='89% HDI')
ax.scatter(y_test, y_pred, marker='o', s=30, c='white', 
           edgecolors='royalblue', label='mean')
ax.legend(loc='upper left')
ax.set_xlabel('Observed EoL')
ax.set_ylabel('Predicted EoL')
ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
axi = fig.add_axes([0.6, 0.26, 0.28, 0.25])
axi.hist(y_test-y_pred, color='lightgrey')
axi.tick_params(which='major', axis='both', labelsize=9)
axi.set_xlabel('Error (cycles)', fontsize=9)
fig.tight_layout()
plt.savefig('residuals.png', dpi=600)
plt.show()

# RMSE and MAPE errors.
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE (test): {rmse:.2f} cycles')
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (test): {mape*100:.2f} %')

# ----------------------------------------------------------------------------
# Multilevel robust multiple linear regression model.
# ----------------------------------------------------------------------------
print('\nMultilevel robust linear model:')
with pm.Model(coords=c) as multi_level:
    # Data containers.
    x_data = pm.Data('data', X_train, dims=('samples', 'features'))
    y_data = pm.Data('y', y_train, dims='samples')
    idx = pm.Data('idx', idx_train, dims='samples')
    # Hyperpriors.
    m = pm.Normal('m', mu=0, sigma=10)
    s = pm.HalfCauchy('s', beta=1)
    # Priors.
    alpha = pm.Normal('alpha', mu=m, sigma=s, shape=2)
    beta = pm.Normal('beta', mu=0, sigma=10, dims='features')
    sigma = pm.HalfStudentT('sigma', nu=4, sigma=1)
    nu = pm.Exponential('nu', lam=1)
    # Linear model with varying intercepts.
    mu = alpha[idx] + x_data @ beta
    # Likelihood.
    obs = pm.StudentT('likelihood', nu=nu, mu=mu, sigma=sigma, 
                      observed=np.log10(y_data), dims='samples')
    # MCMC sampling.
    idata_multi = pm.sample(4000, target_accept=0.95, max_treedepth=15,
                            idata_kwargs={'log_likelihood': True})

# Posterior predictive check.
with multi_level:
    idata_multi.extend(pm.sample_posterior_predictive(idata_multi))

fig, ax = plt.subplots(figsize=(5, 3))
az.plot_ppc(idata_multi, num_pp_samples=500, textsize=11, ax=ax)
ax.set_xlim(2, 3.5)
fig.tight_layout()
plt.savefig('ppc_multi_rlr.png', dpi=600)
plt.show()

# Predict on test data.
with multi_level:
    # Import test data for predictions.
    pm.set_data({'data': X_test}, coords=c_test)
    pm.set_data({'y': y_test}, coords=c_test)
    pm.set_data({'idx': idx_test}, coords=c_test)
    # Predict on test set data.
    y_predict = pm.sample_posterior_predictive(idata_multi, predictions=True)

y_predict_values = 10**y_predict.predictions['likelihood'].mean(dim='chain').values
boxplot_predictions(y_predict_values, y_test,
                    title='Multilevel robust regression (MRLR)', savefig=True)
# Extract mean predicitons.
y_pred = y_predict.predictions['likelihood'].mean(dim=['chain', 'draw']).values
y_pred = 10**y_pred  # return back to targets

# RMSE and MAPE errors.
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE (test): {rmse:.2f} cycles')
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE (test): {mape*100:.2f} %')

# ----------------------------------------------------------------------------
# Bayesian model comparison.
# ----------------------------------------------------------------------------
print('\nModel comparison:')
traces_dict = {
    'OLR': idata,            # ordinary linear regression
    'RLR': idata_robust,     # robust linear regression
    'MRLR': idata_multi,     # multilevel robust linear regression
    'GLM': idata_glm,        # generalized linear model
    'MGLM': idata_multi_glm, # multilevel generalized regression
}
# Pareto-smoothed importance sampling leave-one-out cross-validation (LOOCV).
model_comparison = az.compare(traces_dict, ic='loo', method='stacking')
fig, ax = plt.subplots(figsize=(6, 3.8))
az.plot_compare(model_comparison, plot_ic_diff=True, title=False, 
                textsize=10, plot_kwargs={'color_ic': 'navy'}, ax=ax)
fig.tight_layout()
fig.savefig('comparison.png', dpi=600)
plt.show()

print(az.compare(traces_dict, ic='loo', method='stacking').round(3))
