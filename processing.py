import numpy as np


def import_dataset(filename):
    """ Import Matlab structured data.

    Import a Matlab struct data in a HDF5 format,
    process it and return a structured dictionary.

    Parameters
    ----------
    filename: str
        File name holding the Matlab struct data.
    
    Returns
    -------
    bat_dict: dict
        Nested dictionary holding the data.
    """
    import h5py

    f = h5py.File(filename, mode='r')
    batch = f['batch']
    num_cells = batch['summary'].shape[0]

    bat_dict = {}
    for i in range(num_cells):
        # Summary data.
        cl = f[batch['cycle_life'][i,0]][()]
        Vdlin = f[batch['Vdlin'][i,0]][()].reshape(-1)
        policy = f[batch['policy_readable'][i,0]][()].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i,0]]['IR'][0,:].tolist())
        summary_QC = np.hstack(f[batch['summary'][i,0]]['QCharge'][0,:].tolist())
        summary_QD = np.hstack(f[batch['summary'][i,0]]['QDischarge'][0,:].tolist())
        summary_TA = np.hstack(f[batch['summary'][i,0]]['Tavg'][0,:].tolist())
        summary_TM = np.hstack(f[batch['summary'][i,0]]['Tmin'][0,:].tolist())
        summary_TX = np.hstack(f[batch['summary'][i,0]]['Tmax'][0,:].tolist())
        summary_CT = np.hstack(f[batch['summary'][i,0]]['chargetime'][0,:].tolist())
        summary_CY = np.hstack(f[batch['summary'][i,0]]['cycle'][0,:].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                    summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 
                    'chargetime': summary_CT, 'cycle': summary_CY}
        cycles = f[batch['cycles'][i,0]]
        
        # Data for each cycle.
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j,0]][()]))
            Qc = np.hstack((f[cycles['Qc'][j,0]][()]))
            Qd = np.hstack((f[cycles['Qd'][j,0]][()]))
            Qdlin = np.hstack((f[cycles['Qdlin'][j,0]][()]))
            T = np.hstack((f[cycles['T'][j,0]][()]))
            Tdlin = np.hstack((f[cycles['Tdlin'][j,0]][()]))
            V = np.hstack((f[cycles['V'][j,0]][()]))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j,0]][()]))
            t = np.hstack((f[cycles['t'][j,0]][()]))
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 
                'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}
            cycle_dict[str(j)] = cd
            
        cell_dict = {'Vdlin': Vdlin, 'cycle_life': cl, 'charge_policy':policy, 
                    'summary': summary, 'cycles': cycle_dict}
        key = 'b1c' + str(i)
        bat_dict[key] = cell_dict

    return bat_dict


def filter_signal(data, window_length=20, order=3):
    """
    Remove extreme outliers and then filter the signal.

    First, remove the extreme outliers that are more than
    two standard deviations from the rolling median and then
    smooth the signal by applying the Savitzky-Golay filter.

    Parameters
    ----------
    data: array-like
        Array (1D) holding the raw signal data points.
    window_length: int, default=20
        Window length in sample points for the moving windows,
        and for the Savitzky-Golay filter.
    order: int, default=3
        Polynomial order for the Savitzky-Golay filter.
    
    Returns
    -------
    yhat: array
        Array holding the filtered signal data points. It
        has the same dimension as the original 1D array.
    """
    import pandas as pd
    from scipy.signal import savgol_filter

    s = pd.Series(data, copy=True)
    # Rolling median.
    ma = s.rolling(window=window_length, closed='left').median().values
    # Rolling standard deviation.
    sd = s.rolling(window=window_length, closed='left').std().values
    # Detect outliers that are more than two standard
    # deviations from the rolling median.
    lw = ma - 2*sd
    hi = ma + 2*sd
    
    N = len(data)
    y = np.empty(N)
    for i in range(N):
        if i < window_length:
            # First window.
            y[i] = s.values[i]
        else:
            if lw[i] < s.values[i] < hi[i]:
                y[i] = s.values[i]
            else:
                # Replace an outlier with the associated median value.
                y[i] = ma[i]

    # Smooth data with the Savitzky-Golay filter.
    yhat = savgol_filter(y, window_length, order)

    return yhat


def interpolate_signal(x, y, low, high):
    """
    Linear interpolation of the signal.

    Linear interpolation of the discharge curve for the
    select number of points between starting (`low`) and 
    ending (`high`) cycle numbers.

    Parameters
    ----------
    x: array-like
        Array holding x-coordinate points for the signal.
    y: array-like
        Array holding y-coordinate points for the signal.
    low: int
        Index of the starting point for the approximation.
        This is the first cycle number of the discharge curve
        that will be used for the interpolation.
    high: int
        Index of the ending point for the interpolation.
        This is the last cycle number of the discharge curve
        that will be used for the interpolation.
    
    Returns
    -------
    a0, ai: floats
        Intercept (a0) and slope (ai) of the linear fit 
        through the signal data (between starting and ending 
        points as defined with `low` and `high` parameters).
    """
    from sklearn import linear_model

    lm = linear_model.LinearRegression()
    lm.fit(x[low:high].reshape(-1,1), y[low:high])
    a0 = lm.intercept_  # intercept
    ai = lm.coef_       # slope (list)

    return a0, ai


def get_cell_stats(data, cell_id, cycles, var, full=True):
    """
    Compute statistics for a cell's measurement data. 
    
    Compute different statistics for a measurements 
    array from a number of cycles, for any battery cell 
    from the batch.

    Parameters
    ----------
    data: dict
        Original dictionary with raw measurement data for
        all battery cells and all cycles.
    cell_id: str
        String holding the individual battery cell ID. It
        has a form of 'bxcy', where 'x' is the batch number
        and 'y' is the cell number.
    cycles: int
        Number of cycles for which the data is desired, 
        starting from the first cycle in the dataset.
    var: str
        Variable name for the measured value; it can be one of
        the following: 'I', 'Qc', 'Qd', 'Qdlin', 'T', 'Tdlin', 
        'V', 'dQdV', 't'.
    full: bool, default=True
        Indicator that determines the extent of the statistical
        features that will be computed for the measurement data.
    
    Returns
    -------
    stats_dict: dict
        Dictionary holding various statistics for the measurement 
        data for a single parameter, for the individual cell and 
        the select number of cycles (starting from the second cycle).
    """
    from scipy.stats import mode, skew, kurtosis
    from sklearn.metrics import auc  # area under a curve
    from collections import defaultdict
    
    stats_dict = defaultdict(list)
    support = data[cell_id]['Vdlin']
    for cycle_i in range(2, cycles+1):
        # Retrieve data for each cycle.
        arr = data[cell_id]['cycles'][str(cycle_i)][var]
        # Compute summary statistics for each cycle.
        stats_dict['min'].append(arr.min())
        stats_dict['max'].append(arr.max())
        stats_dict['mean'].append(np.mean(arr))
        stats_dict['std'].append(np.std(arr))
        if full:
            stats_dict['mode'].append(mode(arr)[0])
            stats_dict['skew'].append(skew(arr))  # skewness
            stats_dict['kurt'].append(kurtosis(arr))  # kurtosis (Fisher)
            stats_dict['median'].append(np.median(arr))
            q25 = np.quantile(arr, 0.25)
            q75 = np.quantile(arr, 0.75)
            stats_dict['iqr'] = q75 - q25
            stats_dict['auc'].append(auc(support, arr))  # AUC

    return stats_dict


def get_data_array_stats(data_array, full=True):
    """
    Compute statistics for a single data array. 

    Parameters
    ----------
    data_array: array-like
        Data array.
    full: bool, default=True
        Indicator that determines the extent of the 
        statistical features that will be computed.
    
    Returns
    -------
    stats: dict
        Dictionary holding various statistics for the
        data array.
    """
    from scipy.stats import mode, skew, kurtosis

    stats_dict = {}
    stats_dict['min'] = data_array.min()
    stats_dict['max'] = data_array.max()
    stats_dict['mean'] = np.mean(data_array)
    stats_dict['std'] = np.std(data_array)
    if full:
        stats_dict['mode'] = mode(data_array)[0]
        stats_dict['skew'] = skew(data_array)  # skewness
        stats_dict['kurt'] = kurtosis(data_array)  # kurtosis (Fisher)
        stats_dict['median'] = np.median(data_array)
        q25 = np.quantile(data_array, 0.25)
        q75 = np.quantile(data_array, 0.75)
        stats_dict['iqr'] = q75 - q25

    return stats_dict


def process_multiple_deltas(data, cell_id, var, start, stop, step=1):
    """ 
    Compute mode and AUC for multiple delta curves, for 
    any cell and any measured variable of interest.
    
    Delta curve is obtained by subtracting the i-th curve 
    from the referent curve, e.g., dQ = Qi - Q0, where Q0
    is the referent curve and Qi is a curve from the i-th
    cycle.

    Parameters
    ----------
    data: dict
        Original dictionary with raw measurement data for
        all battery cells and all cycles.
    cell_id: str
        String holding the individual battery cell ID. It
        has a form of 'bxcy', where 'x' is the batch number
        and 'y' is the cell number.
    var: str
        Variable name for the measured value; it can be one of
        the following: 'I', 'Qc', 'Qd', 'Qdlin', 'T', 'Tdlin', 
        'V', 'dQdV', 't'.
    start: int
        Starting (i.e. referent) cycle number.
    stop: int
        Ending (i.e. last) cycle number considered.
    step: int, default=1
        Step value for the cycles between `start` and `stop`.
        By default, each cycle between start and stop will
        be processed.

    Returns
    -------
    modes: array
        Array of modes of the delta curves.
    aucs: array
        Array of AUC (area under the curve) values for the delta curves.
    """
    from scipy.integrate import simpson
    from scipy.stats import mode

    aucs = []
    modes = []
    q0 = data[cell_id]['cycles'][str(1)][var]
    if var == 'Qdlin':
        support = data[cell_id]['Vdlin']
    else:
        support = np.linspace(0, 1, len(q0))

    k = 0
    for i in range(start, stop, step):
        qi = data[cell_id]['cycles'][str(i)][var]
        deltaQ = qi - q0
        # Area under the curve using Simpson's integration.
        aucs.append(simpson(deltaQ, x=support))
        # Assuming no multi-modal curves.
        modes.append(mode(deltaQ)[0])
        k += 1

    return np.asarray(modes), np.asarray(aucs)


def bacon_watts_model(x, alpha0, alpha1, alpha2, x1, gamma=1e-8):
    # Bacon-Watts model for the knee-point identification.
    y = alpha0 + alpha1*(x - x1) + alpha2*(x - x1)*np.tanh((x - x1)/gamma)

    return y


def double_bacon_watts_model(x, alpha0, alpha1, alpha2, alpha3, x0, x2, 
                             gamma=1e-8):
    # Double Bacon-Watts model for the knee-point onset prediction.
    y = alpha0 + alpha1*(x - x0) + alpha2*(x - x0)*np.tanh((x - x0)/gamma) \
        + alpha3*(x - x2)*np.tanh((x - x2)/gamma)

    return y


def fit_bacon_watts_model(x, y, p0, model_type='double'):
    from scipy.optimize import curve_fit
    
    if model_type == 'single':
        # Fit a Bacon-Watts model.
        popt, pcov = curve_fit(bacon_watts_model, x, y, p0=p0)
        # Confidence intervals on the predicted knee-point value.
        confint = [popt[3] - 1.96 * np.diag(pcov)[3], 
                   popt[3] + 1.96 * np.diag(pcov)[3]]
    elif model_type == 'double':
        # Fit a double Bacon-Watts model.
        popt, pcov = curve_fit(double_bacon_watts_model, x, y, p0=p0)
        # Confidence intervals on the predicted knee-point onset value.
        confint = [popt[4] - 1.96 * np.diag(pcov)[4], 
                   popt[4] + 1.96 * np.diag(pcov)[4]]
    else:
        raise NotImplementedError(f'Model {model_type} is not recognized.')
    
    return popt, confint


def get_features_targets_from_data(data_dict, end=100, 
                                   targets='eol', 
                                   skip_outliers=True):
    """
    Extract features and targets from battery cell data.

    Engineer features from cycles and summary measurement
    data for each cell. This includes statistical features
    and discharge fade curve features. Extract also the 
    associated targets for the regression analysis.

    Parameters
    ----------
    data_dict: dict
        Dictionary holding battery cell measurements data.
        This dictionary is formed by importing data.
    end: int, default=100
        Cycle index which marks the end of the observation
        period. All features must be derived from the data
        up to (and including) this cycle number.
    targets: str, default='eol'
        Parameter which defines a type of targets that will
        be returned from the battery cell data. Following
        three values are allowed:
            'eol': End-of-Life values,
            'knee': Knee point values (from the single
                Bacon-Watts model fit),
            'knee-onset': Knee-onset values (from the double
                Bacon-Watts model fit).
    skip_outlier: bool, default=True
        Indicator for skipping outlier battery cells (i.e. 
        those that have 'nan' values for `cycle_life` data
        dictionary keys).

    Returns
    -------
    X_data: dict
        Dictionary holding features for each battery cell.
    y_data: array
        Array holding targets for each battery cell. 
        These depend on the value of the `targets` parameter.
    
    Notes
    -----
    This function features several hard-coded values, which
    are considered defaults, that have been set based on 
    previous research. For example, 2nd cycle is a referent
    starting cycle for many features. These defaults should
    be reviewed and adjusted as necessary.
    
    Important
    ---------
    This function removes (skips) cell records with estimated
    EoL values below 100 cycles, as well as those cells with
    knee and knee-onset point values that were estimated to be
    below 100 cycles. These cells are considered defective.

    References
    ----------
    Kristen A. Severson et al., Data-driven prediction of
    battery cycle life before capacity degradation, Nature
    Energy, Vol. 4, 2019, 383-391, 
    https://doi.org/10.1038/s41560-019-0356-8.
    
    Kristen A. Severson et al., Supplementary Information
    for Data-driven prediction of battery cycle life before 
    capacity degradation, Nature Energy, Vol. 4, 2019.

    P. Fermín-Cueto, et al., Identification and machine
    learning prediction of knee-point and knee-onset in
    capacity degradation curves of lithium-ion cells, Energy
    and AI, Volume 1, 2020, 100006, 
    https://doi.org/10.1016/j.egyai.2020.100006.
    
    P. Fermín et al., Supplementary Information for
    Identification and machine learning prediction of 
    knee-point and knee-onset in capacity degradation curves 
    of lithium-ion cells, Energy and AI, Volume 1, 2020.
    """
    from collections import defaultdict
    from scipy.integrate import simpson
    
    X_data = defaultdict(list)
    y_data_eol = []         # EoL points
    y_data_knee = []        # Knee points
    y_data_knee_onset = []  # Knee-onset points
    # List statistical features of interest. It can be any
    # of the following: 'min', 'max', 'mean', 'std', 'mode', 
    # 'median', 'skew', 'kurt', 'iqr'.
    selected_stats = ['min', 'mean', 'mode', 'std', 'skew', 'kurt', 'iqr']

    for cell in data_dict.keys():
        cycles = data_dict[cell]['summary']['cycle']
        fade_curve = data_dict[cell]['summary']['QD']
        # Smooth the fade discharge curve.
        fade_curve_smooth = filter_signal(fade_curve)
        # End-of-Life cell cycle.
        cycle_life = data_dict[cell]['cycle_life'][0][0]

        if skip_outliers and np.isnan(cycle_life):
            # Skip cell records that have `nan` values
            # for `cycle_life` data dictionary keys.
            print(f'Skipping cell ID: {cell}.')
            continue
        
        if np.isnan(cycle_life):
            # Extract the EoL value.
            curve = fade_curve / fade_curve[0]
            # EoL is at the 80% of initial charge capacity.
            idx = np.argwhere(curve < 0.8)
            if idx.size == 0:
                # Capacity curve did not drop below the 80% margin.
                # Using the last point on the discharge curve as a
                # substitute for the EoL point.
                eol = int(cycles[-1])
            else:
                eol = int(cycles[idx][0])
        else:
            eol = int(cycle_life)
        
        if eol <= 101:
            # Skip cell records that have estimated EoL values
            # below 100 cycles (these are considered defective).
            print(f'Skipping cell ID: {cell} with EoL: {eol}.')
            continue
        if eol < end:
            # EoL is below the observation range.
            raise ValueError(f'Cell ID: {cell} EoL is below the `end` cycle.')
        
        # Targets.
        # Targets are End-of-Life values (default).
        y_data_eol.append(eol)

        # Targets are knee points or knee-onset points.
        if targets == 'knee' or targets == 'knee-onset':
            # Initial values for the fit.
            p0 = [1, -1e-4, -1e-4, 0.7*len(cycles)]
            # Fit a single Bacon-Watts model.
            popt, _ = fit_bacon_watts_model(cycles, fade_curve_smooth, p0,
                                            model_type='single')
            knee_point = int(popt[3])
            # Check the validity of the estimated knee point.
            if targets == 'knee':
                if knee_point < 0:
                    print(f'Cell ID: {cell}, Knee: {knee_point}')
                    raise ValueError(f'Error: {cell} knee point is negative!')
                if knee_point > eol:
                    print(f'Cell ID: {cell}, Knee: {knee_point}, EoL: {eol}')
                    print(f'Warning: {cell} knee-point is beyond the EoL!')
                if knee_point <= 101:
                    # Skip cell records that have estimated knee values
                    # below 100 cycles (these are considered defective).
                    print(f'Skipping cell ID: {cell} with Knee: {knee_point}.')
                    continue
            # Append knee point value.
            y_data_knee.append(knee_point)

            # Initial values for the fit.
            p0 = [popt[0], popt[1] + popt[2]/2, popt[2], popt[2]/2, 
                  0.8*popt[3], 1.1*popt[3]]
            # Fit a double Bacon-Watts model.
            popt_onset, _ = fit_bacon_watts_model(cycles, fade_curve_smooth, p0, 
                                                  model_type='double')
            knee_onset_point = min(int(popt_onset[4]), int(popt_onset[5]))
            # Check the validity of the estimated knee-onset point.
            if targets == 'knee-onset':
                if popt_onset[4] > popt_onset[5]:
                    print('Warning: Issues with a Bacon-Watts fit detected.')
                if knee_onset_point < 0:
                    print(f'Cell ID: {cell}, Knee-onset: {knee_onset_point}')
                    raise ValueError(f'Error: {cell} knee-onset point is negative!')
                if knee_onset_point > eol:
                    print(f'Cell ID: {cell}, Knee-onset: {knee_onset_point}, \
                          EoL: {eol}')
                    print(f'Warning: {cell} knee-onset point is beyond the EoL!')
                if knee_onset_point > knee_point:
                    print(f'Cell ID: {cell}, Knee: {knee_point}, \
                          Knee-onset: {knee_onset_point}')
                    print(f'Warning: {cell} knee-onset point is beyond the Knee point!')
                if knee_onset_point <= 101:
                    # Skip cell records that have estimated knee-onset values
                    # below 100 cycles (these are considered defective).
                    print(f'Skipping cell ID: {cell} with Knee-onset: {knee_onset_point}.')
                    continue
            # Append knee-onset point value.
            y_data_knee_onset.append(knee_onset_point)
                
        # Features.
        voltage = data_dict[cell]['Vdlin']
        # Qd cycles curves.
        Qd10 = data_dict[cell]['cycles'][str(10)]['Qdlin']  # hard-coded
        Qd100 = data_dict[cell]['cycles'][str(end)]['Qdlin']
        # Qd(100) - Qd(10) cycles
        deltaQ = Qd100 - Qd10
        
        # Statistical features from dQ(100-10) curve.
        dQstats = get_data_array_stats(deltaQ)
        for key, value in dQstats.items():
            # Return all calculated statistical properties.
            X_data[key].append(value)
        
        # Discharge capacity fade curve features.
        # Discharge capacity at cycle 2.
        Qd2 = fade_curve_smooth[1]
        if Qd2 > 1.15 or Qd2 < 0.8:
            raise ValueError('Discharge capacity at cycle 2 is out of bounds.')
        X_data['qd2'].append(Qd2)
        # Difference between max discharge capacity and cycle 2.
        X_data['qd_dif'].append(fade_curve_smooth.max() - Qd2)
        # Discharge capacity at cycle 100.
        X_data['qd100'].append(fade_curve_smooth[end-1])

        # Linear fit to the discharge fade curve between cycles 2 and 100.
        intercept, slope = interpolate_signal(cycles, fade_curve_smooth, 2, end)
        X_data['slope'].append(slope[0])
        X_data['inter'].append(intercept)
        
        # Linear fit to the discharge fade curve between cycles 90 and 100.
        interc, slope = interpolate_signal(cycles, fade_curve_smooth, end-10, end)
        X_data['slp9'].append(slope[0])
        X_data['inc9'].append(interc)
        
        # Other features.
        # Average charge time for the first five cycles.
        charge_time = data_dict[cell]['summary']['chargetime']
        X_data['char5'].append(charge_time[1:6].mean())
        
        # Minimum and maximum temperature from cycles 2 to 100.
        # Using the "summary" Tmin and Tmax values.
        t_min = data_dict[cell]['summary']['Tmin']
        t_max = data_dict[cell]['summary']['Tmax']
        X_data['tsmin'].append(t_min[1:end].min())
        X_data['tsmax'].append(t_max[1:end].max())

        # Integral of (average) temperature over time, cycles 2 to 100.
        charge_time = data_dict[cell]['summary']['chargetime']
        temperature = data_dict[cell]['summary']['Tavg']
        t_integral = simpson(temperature[1:end], x=charge_time[1:end])
        X_data['t_int'].append(t_integral)

        # Minimum internal resistance from cycles 2 to 100.
        resistance = data_dict[cell]['summary']['IR']
        X_data['rsmin'].append(resistance[1:end].min())
        # Internal resistance difference between cycles 2 and 100.
        X_data['rsdif'].append(resistance[end-1] - resistance[1])

        # Additional features.
        # Area under the Qd100 - Qd10 discharge curve.
        X_data['dq_auc'].append(simpson(deltaQ, x=voltage))
        
        # dQdV cycles curves.
        dQdV10 = np.diff(Qd10, prepend=0)  # first difference
        dQdV100 = np.diff(Qd100, prepend=0)
        # dQdV(100) - dQdV(10) cycles
        delta_dQdV = dQdV100 - dQdV10
        
        # Statistical features from dQdV(100-10) curve.
        dQdVstats = get_data_array_stats(delta_dQdV)
        for key, value in dQdVstats.items():
            if key in selected_stats:
                # Return only selected stats.
                X_data['dqdv_'+key].append(value)
        
        # Area under the dQdV(100) - dQdV(10) curve.
        X_data['dqdv_auc'].append(simpson(delta_dQdV, x=voltage))

        # Temperature cycles curves.
        Td10 = data_dict[cell]['cycles'][str(10)]['Tdlin']
        Td100 = data_dict[cell]['cycles'][str(end)]['Tdlin']
        # Td(100) - Td(10)
        delta_Td = Td100 - Td10
        
        # Statistical features from Td(100-10) curve.
        td_stats = get_data_array_stats(delta_Td)
        for key, value in td_stats.items():
            if key in selected_stats:
                # Return only selected stats.
                X_data['td_'+key].append(value)
        
        # Area under the Td(100) - Td(10) curve.
        X_data['td_auc'].append(simpson(delta_Td, x=voltage))

        # Average temperature from the first five cycles.
        X_data['tav5'].append(temperature[:5].mean())
        
        # Max absolute difference between Tmin and Tmax cycles 2 to 100.
        t_abs_dif = abs(t_max[1:end] - t_min[1:end])
        X_data['tadif'].append(max(t_abs_dif))

        # Additional features from individual cycles.
        # Discharge curve features from cycles 2 to 50.
        cell_stats_qd = get_cell_stats(data_dict, cell, 50, 'Qdlin')
        # Min. AUC value from the first 50 cycles.
        X_data['auc50q'].append(min(cell_stats_qd['auc']))
        # dQ/dV curve features from cycles 2 to 50.
        cell_stats_dqdv = get_cell_stats(data_dict, cell, 50, 'dQdV')
        # Min. AUC value from the first 50 cycles.
        X_data['auc50qv'].append(min(cell_stats_dqdv['auc']))
    
    # Turn a list of targets into the Numpy array.
    if targets == 'eol':
        # End-of-Life points.
        y_data = np.asarray(y_data_eol)
    elif targets == 'knee':
        # Knee points.
        y_data = np.asarray(y_data_knee)
    elif targets == 'knee-onset':
        # Knee-onset points.
        y_data = np.asarray(y_data_knee_onset)
    else:
        raise NotImplementedError(f'{targets}: Unknown target!')

    return X_data, y_data
