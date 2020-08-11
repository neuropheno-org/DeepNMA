#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:15:22 2020

@author: adonay
"""

import numpy as np
from scipy import interpolate
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.ndimage.measurements import label
from mne.filter import resample
import utils_feature_extraction as feat_ext


def ts_prepro(fingers, timestamp, times=None, max_nans_sec=.5):
    """Prepro fingers TS.
    
    Parameters
    -------------
    fingers : 3D array {finger x (x, y) x n_samples}
        Times series to extract features
    timestamp : n_samples, in seconds
        Time for each sample
    times: array
        indexes where beg and end of TS. If None, ts is segmented when x nans.
    max_nans_sec : Int
        Time in sec of continuous Nans, beyond TS is segmented if times==None.

    Returns
    --------
    fingers_int1: array n_fingers x(x, y) x n_samples
        TS with interpolated nans.
    fingers_int2: array n_fingers x(x, y) x n_samples
        TS with interpolated nans after outlier correction.
    outliers : list fingers [segments [x,y [ix, val]]]
        Outlier indexes and values
    """

    Fs = 1/ np.average(np.diff(timestamp[:,0]))
    max_nans = max_nans_sec*Fs

    fingers_int1 = np.empty(shape=fingers.shape)
    fingers_int1[:] = np.nan
    fingers_int2 = np.empty(shape=fingers.shape)
    fingers_int2[:] = np.nan
    
    outliers = []
    for i_f, finger in enumerate(fingers):
        if times is None or times.shape[0]==0:
            ts_segments, seg_ix = get_continuous_ts(finger, max_nans)
        else:  # Use times for seg Ts
            if np.any(np.isnan(times)):
                ts_segments, seg_ix = [], []
                crx
            else:
                times = times.astype("int")
                # Segements Ts based on times (right or left hand)
                ts_segments = [finger[:, times[0]: times[1]] 
                               if i_f < fingers.shape[0]/2 else
                               finger[:, times[2]: times[3]]]

                seg_ix = [np.arange(times[0], times[1])
                          if i_f < fingers.shape[0]/2 else
                          np.arange(times[2], times[3])]

        s_out = []
        for s_ix, s_ts in zip(seg_ix, ts_segments):
            msk = np.isnan(s_ts[0,:])
            s_tstm = timestamp[s_ix,0]
            
            s_ts_int1, _ = interp_tracking(s_ts, s_tstm, msk, lin_space=False)
            o_ix, o_val, _ = detect_outlier(s_ts_int1)
            # get outlier ts time, index and value 
            s_otl = [np.vstack((s_ix[i], v)) for i, v in zip(o_ix, o_val)]
            s_out.append(s_otl)

            s_ts_into = correct_outliers(s_ts_int1, o_ix, o_val)
            s_ts_int2, _ = interp_tracking(s_ts_into, s_tstm, msk,
                                           lin_space=False)
            # Rebuild ts corrected
            fingers_int1[i_f, :, s_ix] = s_ts_int1.T
            fingers_int2[i_f, :, s_ix] = s_ts_int2.T
        # Outliers with shape finger:(x,y):[ix, val]
        o =[[np.hstack([i[n] for i in s_out ])] for n in range(2)]
        outliers.append(o)

    return fingers_int1, fingers_int2, outliers


def get_continuous_ts(finger, max_nans=30):
    """ Split TS where n consecutive nans bigger than max_nans. 1D only"""
    gaps = np.isnan(finger)
    assert(np.all(gaps[0,:] == gaps[1,:])), "Finger x, y dim not equal. Check!"
    # Use one axis (x,y)
    gaps = gaps[0,:]
    seg, n_seg  = label(gaps)

    for s in range(1, n_seg+1):
        if len(seg[seg==s]) < max_nans:
            gaps[seg==s] = False

    fing_seg, seg_ix = [], []
    seg, n_seg  = label(gaps==0)
    for s in range(1, n_seg+1):
        ix, = np.where(seg==s)
        s_ts, ix = _remove_nans_ts(finger[np.newaxis, :, ix], ix)
        if np.sum(~np.isnan(s_ts[0,0,:]))<4:
            continue
        s_ts = np.squeeze(s_ts)
        fing_seg.append(s_ts)
        seg_ix.append(ix)

    return fing_seg, seg_ix


def resample_ts(ts, time, sfreq, sfreq_common):
    """Resample signal to a common sfreq."""
    TS_rs, time_rs = [], []
    rtio = sfreq/sfreq_common
    if rtio < 1:
        TS_rs = resample(ts, up=rtio)
    elif rtio > 1:
        TS_rs = resample(ts, down=rtio)

    time_rs = np.arange(time[0], time[-1]+1, 1/sfreq_common) 
    time_rs = time_rs[:TS_rs.size]
    return TS_rs, time_rs, rtio


def interp_tracking(TS, timestamp, mask, lin_space):
    """Interpolates nans in time series and int. into evenly spaced time.
    
    Parameters
    -------------
    TS : array (x, y) x n_samples
        Times series to extract features
    timestamp : n_samples, in seconds
        Time for each sample
    mask: array n_samples
        indexes where NaNs are present
    lin_space : bool
        If True, interpolate TS in linearly spaced time

    Returns
    --------
    TS_int: array (x, y) x n_samples
        TS with interpolated nans and time.
    time: n_samples, in seconds
        New time linearly spaced if times_true is True.
    """

    mask_gd = ~mask
    times_true = timestamp[mask_gd]
    if lin_space is True:
        time = np.linspace(times_true[0], times_true[-1], len(timestamp))
    else:
        time = timestamp

    TS_int = []
    for ts in TS:
        fcubic = interpolate.interp1d(times_true.T, ts[mask_gd], kind='cubic')
        TS_int.append(fcubic(time))

    return np.vstack(TS_int), time

def _remove_nans_ts(fingers_cut, times_stm):
    # First and last sample cannot be a NaN
    fst_smp, lst_smp = 0, fingers_cut.shape[-1]
    nans = np.any(np.isnan(fingers_cut),(0,1))
    if nans[0]:
        fst_smp = np.where(nans==False)[0][0] # get first non Nan
    if nans[-1]:
        lst_smp =  np.where(nans==False)[0][-1] + 1 # get first non Nan)
    inx_nonan = np.arange(fst_smp, lst_smp)
    fingers_cut, times_stm = fingers_cut[:,:,inx_nonan], \
        times_stm[inx_nonan]
    return fingers_cut, times_stm


def detect_outlier(TS, samples_wind=60, order=3):
    """Find outliers in TS by interpolate one sample at a time, measure diff.
    between rec. sample and interpolated, and getting the peaks in the int diff
    across recording. 
    
    Parameters
    -------------
    TS : array (x, y) x n_samples
        Times series to extract features
    samples_wind : int
        Window length of segment where a sample is interpolated.
    order : int
        B-sline interpolation order

    Returns
    --------
    outliers: list of array n_chans [n_outliers]
        Indices of outliers per chans
    outliers_int: list of array n_chans [n_outliers]
        New interpolated values of the  outliers
    """

    s_win_half = int(samples_wind/2)
    outliers = []
    outliers_int = []
    zdiffs = []
    for ts in TS:
        n_samples, = ts.shape
        diff = [np.nan]
        ts_int_one = [np.nan]
        for w in range(1,n_samples-1):
            wix = [w-s_win_half,w+s_win_half]
            # Bound beg or end if outside
            wix[0] = 0 if wix[0]<0 else wix[0]
            wix[1] = n_samples if wix[1]>n_samples else wix[1]

            seg1, seg2 = ts[wix[0]:w], ts[w+1:wix[1]]
            seg = np.concatenate((seg1,seg2))
            # make indexes ts with and without sample
            ixs = np.arange(seg.shape[0]+1)
            ixs_out =np. delete(ixs, np.argwhere(ixs == seg1.shape[0]))
            # Interpolate and measure diff
            fcubic = interpolate.interp1d(ixs_out, seg, kind=order)
            ts_int_out = fcubic(ixs)
            smpl_int = ts_int_out[seg1.shape[0]]
            diff.append(np.abs(smpl_int-ts[w]))
            ts_int_one.append(smpl_int)
            
        diff_z = zscore(diff)
        pks_p, _ = feat_ext.find_maxmin_peaks(diff_z[1:], height=3.5)
        pks_p = pks_p + 1  # add 1 sampl ( first is nan)
        int_smp = np.array(ts_int_one)[pks_p]

        outliers.append(pks_p)
        outliers_int.append(int_smp)
        zdiffs.append(diff_z)
    return outliers, outliers_int, np.array(zdiffs)


def get_dist_nans(fingers, int2, max_nans):
    
    fing_nans = []
    for i, (raw, intrp) in enumerate(zip(fingers, int2)):
        # Get nans from interp TS
        no_nns, = np.where(np.isnan(intrp[1]) == 0)
        t0, t1 = no_nns[0], no_nns[-1]
        # Get nans from raw TS within task
        nns, = np.where(np.isnan(raw[1]))
        fing_nans.extend(nns[(nns>t0) & (nns < t1)])

    n_nans = len(fing_nans)
    if n_nans <= max_nans:
        return fing_nans

    # Cluster position get max_nans clusters
    coord = int2[:,:, fing_nans]

    # Binary flat image for clustering
    d1, d2 = np.ceil(np.nanmax(coord[:,0,:])), np.ceil(np.nanmax(coord[:,1,:]))
    img = np.zeros((int(d1), int(d2), n_nans))
    for i in range(n_nans):
        for cord in coord[:, :, i]:
            if not np.isnan(cord).all():
                cord = cord.astype('int')
                img[cord[0], cord[1], i] = 1

    img_f = np.reshape(img, (-1, n_nans)).T

    # Cluster and get closest image inx
    kmeans = KMeans(n_clusters=max_nans).fit(img_f)
    k_cent =  kmeans.cluster_centers_
    closest, _ = pairwise_distances_argmin_min(k_cent, img_f)
    return [fing_nans[c] for c in closest]

def correct_outliers(ts, outliers, outliers_int):
    ts_o = ts.copy()
    for i, (o, oi) in enumerate(zip(outliers, outliers_int)):
        if len(o):
            ts_o[i,o] = oi
    return ts_o

def restore_bad_outliers(out_checked, int1, int2):
    new_int2 = int2.copy()
    for out in out_checked:
        if 'bad' in out:
            a, b, c = [int(i) for i in out[:3]]
            new_int2[a, b, c] = int1[a, b, c]

    return new_int2

def zscore(x):
    x = (x - np.nanmean(x))/ np.nanstd(x)
    return x

def data_augmentation(ts, wind_len=150, n_wind_ratio=5):
    """Splits data into `wind_len` * n_windows based on the original ts length
    and the `wind_len` ratio. Then every three windows the first wind is the
    original windowed ts, the second wind is flipped, the third is upsampled
    and cropped to `wind_len`. if `ts` is too short, a copy is returned with
    no augmentation.
    
    Parameters
    -------------
    ts : array 1D x n_samples
        Times series to augment
    wind_len : int
        Window length of segment.
    n_wind_ratio : int
        The ratio ts_len/wind_len is multiplid by this parameter
    Returns
    --------
    ts_augm: array n_samples x n_wind
        New augmented data/
    """

    ts_shp = ts.shape[0]

    if ts_shp > wind_len:
        ratio = ts_shp / wind_len
        n_winds = round(ratio*n_wind_ratio)
        # get points of window start
        points = np.arange(0, ts_shp-wind_len)
        points = points[:: round(points.size/n_winds)]

        ts_augm = []
        for p in points[::3]: # Sliding
            ts_augm.append(ts[p:p+wind_len])
        # for p in points[1::3]: # Flipping + 1 sample
        #     ts_augm.append(ts[p + wind_len:p:-1])
        for p in points[2::3]: # upsampling
            ts_sub = ts[p:p+wind_len]
            ts_sub_rs = resample(ts_sub, up=2)
            ts_beg = round(ts_sub_rs.size/4)
            ts_augm.append(ts_sub_rs[ts_beg:ts_beg + wind_len])

        ts_augm = np.stack(ts_augm,1)

    else:
         t0,t1 = int(ts_shp/2) , int(wind_len/2)
         ts_augm = np.atleast_2d(ts[t0 - t1 : t0 + t1]).T

    return ts_augm