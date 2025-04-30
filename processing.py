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

    First, remove the extreme outliers using the median
    filtering technique and then further smooth the signal
    by applying the Savitzky-Golay filter.

    Parameters
    ----------
    data: array-like
        Array (1D) holding the raw signal data points.
    window_length: int, default=20
        Window length in sample points for the median filter
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
    from scipy.signal import medfilt, savgol_filter

    s = pd.Series(data, copy=True)
    if window_length % 2 == 0:
        window_length += 1
    
    # Median filtering.
    sf = medfilt(s, kernel_size=window_length)

    # Smooth data with the Savitzky-Golay filter.
    yhat = savgol_filter(sf, window_length, order)

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


def get_cell_stats(data, cell_id, cycles, var, window_size=21,
                   full=True, THR=10):
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
    window_size: int, default=21
        Window size for the median filtering of the signal. It
        mast be an odd value.
    full: bool, default=True
        Indicator that determines the extent of the statistical
        features that will be computed for the measurement data.
    THR: float, default=10
        Bound on the legal values of the measurement data.
    
    Returns
    -------
    stats_dict: dict
        Dictionary holding various statistics for the measurement 
        data for a single parameter, for the individual cell and 
        the select number of cycles (starting from the second cycle).
    """
    from scipy.stats import mode, skew, kurtosis
    from scipy.signal import medfilt
    from sklearn.metrics import auc  # area under a curve
    from collections import defaultdict
    
    stats_dict = defaultdict(list)
    support = data[cell_id]['Vdlin']
    for cycle_i in range(2, cycles+1):
        # Retrieve raw data for the cycle.
        arr_raw = data[cell_id]['cycles'][str(cycle_i)][var]
        # Apply median filtering.
        if window_size % 2 == 0:
            window_size += 1  # must be odd value
        arr = medfilt(arr_raw, kernel_size=window_size)
        # Test against bounds on legal values.
        if np.any(abs(arr) > THR):
            print(f'Cell ID: {cell_id} cycle {cycle_i} {var} data out of range.')
            # Skip this record.
            continue

        # Compute summary statistics for the cycle.
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
                                   skip_outliers=True):
    """
    Extract features and targets from battery cell data.

    Engineer features from cycles and summary measurement
    data for each cell. This includes statistical features
    and discharge fade curve features. Extract also the 
    associated targets for the regression or classification 
    analysis.

    Parameters
    ----------
    data_dict: dict
        Dictionary holding battery cell measurements data.
        This dictionary is formed by importing data.
    end: int, default=100
        Cycle index which marks the end of the observation
        period. All features must be derived from the data
        up to (and including) this cycle number.
    skip_outlier: bool, default=True
        Indicator for skipping outlier battery cells (i.e. 
        those that have 'nan' values for `cycle_life` data
        dictionary keys).

    Returns
    -------
    X_data: dict
        Dictionary holding features for each battery cell.
    y_data: dict
        Dictionary holding targets for each battery cell, 
        with following four keys:
            'eol': End-of-Life values,
            'knee': Knee point values (from the single
                Bacon-Watts model fit),
            'knee-onset': Knee-onset values (from the double
                Bacon-Watts model fit).
            'class': classification labels (0 - long life cell,
                1 - short life cell, where life threshold has
                been set at 550 cycles).
        First three target types are used for regression and 
        the fourth target is used for classification.
    
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
    y_data_class = []       # classification labels
    # List statistical features of interest. It can be any
    # of the following: 'min', 'max', 'mean', 'std', 'mode', 
    # 'median', 'skew', 'kurt', 'iqr'.
    selected_stats = ['min', 'mean', 'mode', 'std', 'skew', 'kurt', 'iqr']
    
    THRESHOLD = 10.  # Bound on Qd curve legal values.
    for cell in data_dict.keys():
        cycles = data_dict[cell]['summary']['cycle']
        fade_curve = data_dict[cell]['summary']['QD']
        # Smooth the fade discharge curve.
        fade_curve_smooth = filter_signal(fade_curve, order=2)
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

        # Classification targets (class labels).
        short_life_threshold = 550
        if eol <= short_life_threshold:
            # Short life cell.
            klasa = 1
        else:
            # Long life cell.
            klasa = 0
        y_data_class.append(klasa)

        # Targets are knee points.
        # Initial values for the fit.
        p0 = [1, -1e-4, -1e-4, 0.7*len(cycles)]
        # Fit a single Bacon-Watts model.
        popt, _ = fit_bacon_watts_model(cycles, fade_curve_smooth, p0,
                                        model_type='single')
        knee_point = int(popt[3])
        
        # Check the validity of the estimated knee point.
        if knee_point < 0:
            print(f'Cell ID: {cell}, Knee: {knee_point}')
            raise ValueError(f'Error: {cell} knee point is negative!')
        if knee_point > eol:
            print(f'Cell ID: {cell}, Knee: {knee_point}, EoL: {eol}')
        if knee_point <= 101:
            # Skip cell records that have estimated knee values
            # below 100 cycles (these are considered defective).
            print(f'Skipping cell ID: {cell} with Knee: {knee_point}.')
            continue
        
        # Append knee point value.
        y_data_knee.append(knee_point)
        
        # Targets are knee-onset points.
        # Initial values for the fit.
        p0 = [popt[0], popt[1] + popt[2]/2, popt[2], popt[2]/2, 
                0.8*popt[3], 1.1*popt[3]]
        # Fit a double Bacon-Watts model.
        popt_onset, _ = fit_bacon_watts_model(cycles, fade_curve_smooth, p0, 
                                                model_type='double')
        knee_onset_point = min(int(popt_onset[4]), int(popt_onset[5]))
        
        # Check the validity of the estimated knee-onset point.
        if popt_onset[4] > popt_onset[5]:
            print('Warning: Issues with a Bacon-Watts fit detected.')
        if knee_onset_point < 0:
            print(f'Cell ID: {cell}, Knee-onset: {knee_onset_point}')
            raise ValueError(f'Error: {cell} knee-onset point is negative!')
        if knee_onset_point > eol:
            print(f'Cell ID: {cell}, Knee-onset: {knee_onset_point}, \
                    EoL: {eol}')
        if knee_onset_point > knee_point:
            print(f'Cell ID: {cell}, Knee: {knee_point}, \
                    Knee-onset: {knee_onset_point}')
        if knee_onset_point <= 101:
            # Skip cell records that have estimated knee-onset values
            # below 100 cycles (these are considered defective).
            print(f'Skipping cell ID: {cell} with Knee-onset: {knee_onset_point}.')
            continue
        
        # Append knee-onset point value.
        y_data_knee_onset.append(knee_onset_point)
                
        # Classification features.
        Qd4 = data_dict[cell]['cycles'][str(4)]['Qdlin']  # hard-coded
        Qd5 = data_dict[cell]['cycles'][str(5)]['Qdlin']  # hard-coded
        if np.any(abs(Qd4) > THRESHOLD):
            raise ValueError(f'Cell ID: {cell} Qd4 cycle values out of range.')
        if np.any(abs(Qd5) > THRESHOLD):
            raise ValueError(f'Cell ID: {cell} Qd5 cycle values out of range.')
        
        # Qd(5) - Qd(4) cycles.
        deltaq = Qd5 - Qd4
        
        # Statistical features from dQ(5-4) curve.
        dq_stats = get_data_array_stats(deltaq)
        for key, value in dq_stats.items():
            if key in ['min', 'std']:
                X_data['class_'+key].append(value)

        # Regression features.
        voltage = data_dict[cell]['Vdlin']
        
        # Qd cycles curves.
        Qd10 = data_dict[cell]['cycles'][str(10)]['Qdlin']  # hard-coded
        Qd100 = data_dict[cell]['cycles'][str(end)]['Qdlin']
        if np.any(abs(Qd10) > THRESHOLD):
            raise ValueError(f'Cell ID: {cell} Qd(10) cycle values out of range.')
        if np.any(abs(Qd100) > THRESHOLD):
            raise ValueError(f'Cell ID: {cell} Qd(100) cycle values out of range.')
        
        # Qd(100) - Qd(10) cycles.
        deltaQ = Qd100 - Qd10
        
        # Statistical features from dQ(100-10) curve.
        dQstats = get_data_array_stats(deltaQ)
        for key, value in dQstats.items():
            if key in selected_stats:
                # Return only selected stats.
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
        dQdV10 = data_dict[cell]['cycles'][str(10)]['dQdV']
        dQdV100 = data_dict[cell]['cycles'][str(end)]['dQdV']
        
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

        # Additional features from multiple individual cycles.
        # Discharge curve features from cycles 2 to 50.
        cell_stats_qd = get_cell_stats(data_dict, cell, 50, 'Qdlin', 
                                       window_size=20, THR=THRESHOLD)
        # Min. AUC value from the first 50 cycles.
        X_data['auc50q'].append(min(cell_stats_qd['auc']))
        # Abs. difference between AUC values, cycles 2 and 50.
        X_data['auc_d'].append(abs(cell_stats_qd['auc'][-1]
                                   - cell_stats_qd['auc'][1]))
        # Intercept and slope of the line fit through the Qd
        # evolution of AUC values from the first 50 cycles.
        support = np.arange(len(cell_stats_qd['auc']))
        a0, ai = interpolate_signal(support, cell_stats_qd['auc'], 0, -1)
        X_data['auc_a0'].append(a0)
        X_data['auc_ai'].append(ai[0])
        
        # Process cycles from Qd(2) to Qd(50).
        modes, aucs = process_multiple_deltas(data_dict, cell, 'Qdlin', 2, 50)
        X_data['mod50'].append(modes[-1])  # Mode from Qd(50) - Qd(2)
        X_data['auc50'].append(aucs[-1])   # AUC from Qd(50) - Qd(2)

        # dQ/dV curve features from cycles 2 to 50.
        cell_stats_dqdv = get_cell_stats(data_dict, cell, 50, 'dQdV', 
                                         window_size=20, THR=THRESHOLD)
        # Min. AUC value from the first 50 cycles.
        X_data['auc50qv'].append(min(cell_stats_dqdv['auc']))
        # Abs. difference between AUC values, cycles 2 and 50.
        X_data['auc_dqv'].append(abs(cell_stats_dqdv['auc'][-1]
                                   - cell_stats_dqdv['auc'][1]))
        # Abs. difference between modes, cycles 2 and 50.
        X_data['md_qv'].append(abs(cell_stats_dqdv['mode'][-1] 
                                   - cell_stats_dqdv['mode'][1]))
        # Intercept and slope of the line fit through the dQdV
        # evolution of AUC values from the first 50 cycles.
        support = np.arange(len(cell_stats_dqdv['auc']))
        a0, ai = interpolate_signal(support, cell_stats_dqdv['auc'], 0, -1)
        X_data['auc_a0qv'].append(a0)
        X_data['auc_aiqv'].append(ai[0])
    
    # Dictionary of target values.
    y_data = {}
    # End-of-Life points.
    y_data['eol'] = np.asarray(y_data_eol)
    # Knee points.
    y_data['knee'] = np.asarray(y_data_knee)
    # Knee-onset points.
    y_data['knee-onset'] = np.asarray(y_data_knee_onset)
    # Classification labels.
    y_data['class'] = np.asarray(y_data_class)

    return X_data, y_data


def cluster_membership(X_data, metric='euclidean', 
                       graph_structure=False, n_neighbors=5, 
                       linkage='ward'):
    """
    Cluster analysis on top of the features.

    Agglomerative lustering is appled to features in order
    to separate battery cells into two independent clusters.
    Hopefully, clustering will identify battery cell samples 
    with short and long life and differentiate between them.

    Parameters
    ----------
    X_data: DataFrame
        Pandas DataFrame holding selected features for 
        each battery cell.
    metric: str, default='euclidean'
        Metric used to compute the linkage. Can be 
        'euclidean', 'l1', 'l2', 'manhattan', 'cosine'.
    graph_structure: bool, default=False
        Generate connectivity matrix from the k-neighbors
        graph analysis.
    n_neighbors: int, default=5
        Number of neighbors for each sample, for the
        k-neighbors graph analysis.
    linkage: str, default='ward'
        Linkage criterion to use. Can be 'ward', 
        'complete', 'average', 'single'.
        
    Returns
    -------
    cluster_labels: list
        List of cluster membership labels for each 
        battery cell sample.
    """
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import AgglomerativeClustering

    if graph_structure:
        graph = kneighbors_graph(X_data.values, n_neighbors=n_neighbors, 
                                 metric=metric, n_jobs=-1)
    else:
        graph = None
    
    # Perform an agglomerative clustering. 
    clustering = AgglomerativeClustering(
        n_clusters=2,
        metric=metric,
        connectivity=graph,
        linkage=linkage
    )
    clustering.fit(X_data.values)
    
    # Extract cluster memberships.
    cluster_labels = clustering.labels_

    return cluster_labels


def prepare_features(X_data_dict, set_of_features, 
                     cluster=False, **kwargs):
    """
    Prepare features for the regression models.

    Parameters
    ----------
    X_data: dict
        Dictionary holding all engineered features.
    set_of_features: str
        Designation for the set of features that will
        be used for the model. Following values are
        allowed for this parameter: 'variance',
        'discharge', 'full', 'custom', 'all'.
    cluster: bool, default=False
        Apply clustering analysis on top of the features.
    kwargs: dict
        Additional parameters passed down to the clustering
        function.
        
    Returns
    -------
    X_data: pd.DataFrame
        Pandas dataframe holding selected features.
    """
    import pandas as pd

    if set_of_features == 'variance':
        # Using a single feature.
        X_data = pd.DataFrame(X_data_dict, columns=['std'])
        X_data['std'] = np.log10(X_data['std'].values**2)

    elif set_of_features == 'discharge':
        selected_features = ['std', 'min', 'skew', 'kurt', 'qd2', 'qd_dif']
        X_data = pd.DataFrame(X_data_dict, columns=selected_features)

    elif set_of_features ==  'full':
        selected_features = ['std', 'min', 'qd2', 'slope', 'inter', 
                             'char5', 't_int', 'rsmin', 'rsdif']
        X_data = pd.DataFrame(X_data_dict, columns=selected_features)

    elif set_of_features == 'custom':
        selected_features = ['min', 'std', 'kurt', 'iqr', 
                             'dq_auc', 'auc50q', 'auc_d', 'auc_ai']
        X_data = pd.DataFrame(X_data_dict, columns=selected_features)

    elif set_of_features == '50cycles':
        selected_features = ['auc50q', 'auc_d', 'auc_ai', 'auc50qv', 
                             'auc_dqv', 'auc_aiqv', 'mod50', 'auc50']
        X_data = pd.DataFrame(X_data_dict, columns=selected_features)

    elif set_of_features == 'all':
        # Using all derived features.
        X_data = pd.DataFrame(X_data_dict)
        # Remove classification features.
        X_data.drop(columns=['class_min', 'class_std'], inplace=True)

    else:
        raise NotImplementedError(f'Model: {set_of_features} is not recognized!')
    
    if cluster:
        # Perform clustering on selected features and
        # add cluster memberships as a new feature.
        X_data['cluster'] = cluster_membership(X_data.copy(), **kwargs)

    return X_data


def get_grid_values(reg):
    """
    Create a fixed grid of hyperparameter values.

    Parameters
    ----------
    reg: str
        Short code for the model type. Following 
        values are allowed: 
        'lin': Penalized linear regression,
        'gen': Generalized linear regression with 
               a Tweedie distribution,
        'nusvr': Nu-Support Vector Machine regression,
        'svr': Support Vector Machine regression,
        'ard': Relevance Vector Machine regression.

    Returns
    -------
    grid: dict
        Dictionary with model parameters names as keys 
        and lists of parameter settings to try as values.
    """
    # Grid of hyperparameters values.
    n_values = 10
    if reg == 'lin':
        # ElasticNet
        grid = {'alpha': np.logspace(-3, 3, n_values),
                'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]}
    elif reg == 'gen':
        # TweedieRegressor
        grid = {'power': [0, 1.5, 2, 3],
                'alpha': np.logspace(-3, 3, n_values)}
    elif reg == 'nusvr':
        # NuSVR
        grid = {'nu': [.1, .3, .5, .7, .9, .95, .99],
                'gamma': ['scale', 'auto'],
                'C': np.logspace(-3, 3, n_values)}
    elif reg == 'svr':
        # SVR
        grid = {'gamma': ['scale', 'auto'],
                'C': np.logspace(-3, 3, n_values)}
    elif reg == 'ard':
        # ARDRegression
        grid = {'alpha_1': np.logspace(-5, 1, n_values),
                'alpha_2': np.linspace(0, 1, n_features),
                'lambda_1': np.logspace(-5, 1, n_values),
                'lambda_2': np.linspace(0, 1, n_features)}
    else:
        raise NotImplementedError(f'Chosen model {reg} is not implemented.')

    return grid


def run_regression(reg, X, y, train_size=0.8, 
                   reduce_features=False, n_features=6,
                   verbose=False):
    """
    Run a regression model.

    This function encapsulates the complete pipeline,
    from train and test data split, grid search with
    cross-validation for model hyperparameters optimi-
    zation, fiting the optimal model on the train set 
    and predicting on test data set, to finally compu-
    ting the RMSE and MAPE errors on the test set.

    Parameters
    ----------
    reg: str
        Short code for the model type. Following 
        values are allowed: 
        'lin': Penalized linear regression,
        'gen': Generalized linear regression with 
               a Tweedie distribution,
        'nusvr': Nu-Support Vector Machine regression,
        'svr': Support Vector Machine regression,
        'ard': Relevance Vector Machine regression.
    X: 2d-array
        Matrix of features (predictors).
    y: 1d-array
        Vector of targets (predicted variable).
    train_size: float, default=0.8
        Size of the train set (0, 1).
    reduce_features: bool, default=False
        Indicator for activating fetaures reduction
        option of the data processing pipeline.
    n_features, int, default=6
        Number of features to retain after the features
        reduction process.
    verbose: bool, default=False
        Print diagnostic information.
    
    Returns
    -------
    rmse: float
        Root mean squared error on the test set.
    mape: float
        Mean absolute percentage error on the test set.
    """
    from sklearn import svm
    from sklearn.linear_model import ElasticNet, TweedieRegressor, ARDRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import RFE
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import GridSearchCV

    if reduce_features and n_features > len(X.columns):
            raise ValueError(f'Cannot reduce features with: {n_features} > {len(X.columns)}!')
    
    # Split dataset into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, shuffle=True
    )
    # List of numerical features.
    num_features = [name for name in X.columns if name != 'cluster']
    
    if reg == 'lin':
        # Penalized linear regression.
        reg_model = ElasticNet(max_iter=10_000)
    elif reg == 'gen':
        # Generalized linear regression with a Tweedie distribution.
        reg_model = TweedieRegressor(max_iter=1000)
    elif reg == 'nusvr':
        # Nu-Support Vector Machine regression.
        reg_model = svm.NuSVR(kernel='rbf')
    elif reg == 'svr':
        # Support Vector Machine regression.
        reg_model = svm.SVR(kernel='rbf')
    elif reg == 'ard':
        # Relevance Vector Machine regression.
        reg_model = ARDRegression()
    else:
        raise NotImplementedError(f'Chosen model {reg} is not implemented.')

    # Get a fixed grid of hyperparameter values.
    grid = get_grid_values(reg)

    # Hyperparameters optimization with grid search and cross-validation.
    grid_search = GridSearchCV(estimator=reg_model, param_grid=grid, 
                               scoring='neg_root_mean_squared_error', 
                               cv=3, refit=True, n_jobs=-1)
    # Pipeline.
    if reduce_features:
        model = Pipeline([
            # Transform only numerical features.
            ('transform', ColumnTransformer(
                [('scale', StandardScaler(), num_features)], 
                remainder='passthrough')),
            # Recursive features elimination.
            ('select', RFE(estimator=svm.SVR(kernel='linear'), 
                           n_features_to_select=n_features)),
            ('model', grid_search)
        ])
    else:
        model = Pipeline([
            # Transform only numerical features.
            ('transform', ColumnTransformer(
                [('scale', StandardScaler(), num_features)], 
                remainder='passthrough')),
            ('model', grid_search)
        ])

    # Model fit on train set.
    model.fit(X_train, np.log10(y_train))
    if verbose:
        print(model['model'].best_params_)

    # Predicted values.
    y_pred = model.predict(X_test)
    y_pred = 10**y_pred  # return back to targets

    # Errors from the test set.
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return rmse, mape*100.


if __name__ == '__main__':
    import os
    import pickle

    # Input parameters:
    # ----------------
    targets = 'eol'  # ['eol', 'knee', 'knee-onset']
    set_of_features = 'custom'  # ['variance', 'discharge', 'full', 'custom', 'all']
    reg = 'nusvr'  # ['lin', 'gen', 'nusvr', 'svr', 'ard']
    # Additional options:
    is_pickle = True     # Data format (pickle or matlab).
    reduce_features = False  # Reduce nmber of features.
    n_features = 6    # Number of features to retain.
    verbose = False   # Print optimal hyperparameters.
    cluster = True   # Use clustering analysis on features.

    print('Importing data ...')
    # Import data from the external file.
    path = os.path.dirname('/home/ps/Documents/Batteries/Data/')  # EDIT HERE
    file_name = '2018-02-20_batchdata_updated_struct_errorcorrect'

    if is_pickle:
        # Import data from a pickle file (faster).
        pickle_file = file_name+'.pkl'
        file_path = os.path.join(path, pickle_file)
        fp = open(file=file_path, mode='rb')
        bat_dict = pickle.load(fp)
        fp.close()
    else:
        # Import data from a matlab file (slower).
        matlab_file = file_name+'.mat'
        file_path = os.path.join(path, matlab_file)
        bat_dict = import_dataset(file_path)
        # Export data into a pickle file.
        fp = open(file=file_name+'.pkl', mode='wb')
        pickle.dump(bat_dict, file=fp)
        fp.close()
    
    print('Extracting features ...')
    # Extract features from data.
    X_data_dict, y_data_dict = get_features_targets_from_data(
        bat_dict, skip_outliers=False
    )
    # Prepare features.
    X_data = prepare_features(X_data_dict, set_of_features, cluster=cluster)

    # Prepare targets.
    y_data = y_data_dict[targets]
    
    print('\nRunning model ...')
    if reg == 'lin':
        model_name = 'ElasticNet'
    elif reg == 'gen':
        model_name = 'TweedieRegressor'
    elif reg == 'nusvr':
        model_name = 'NuSVR'
    elif reg == 'svr':
        model_name = 'SVR'
    elif reg == 'ard':
        model_name = 'ARD'
    else:
        raise NotImplementedError(f'Model {reg} is not recognized.')

    # Running the regression analysis multiple times 
    # to acquire summary performance statistics.
    num_runs = 5
    rmse, mape = [], []
    for i in range(num_runs):
        r, m = run_regression(reg, X_data, y_data, 
                              reduce_features=reduce_features, 
                              n_features=n_features, 
                              verbose=verbose)
        rmse.append(r)
        mape.append(m)

    print(f'\nPredicting "{targets}" using "{set_of_features}" set of features.')
    print(f'Test errors from {num_runs} runs ({model_name}):')
    print(f'RMSE: {np.mean(rmse):.2f} +/- {np.std(rmse):.2f} cycles')
    print(f'MAPE: {np.mean(mape):.2f} +/- {np.std(mape):.2f} %')
