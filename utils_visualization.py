#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:27:21 2020

@author: adonay
"""
import numpy as np
import matplotlib.pyplot as plt
import utils_io as uio

## Plotting functions

def plot_per_axis(ts, time, color, fig_axes, **kwargs):
    "For each axes subplot plots ts and time"
    for i, axis in enumerate(fig_axes):
        axis.plot(time, ts[:, i, :].T, color,**kwargs)


def plot_per_axis_outlier(outliers, time, color, fig_axes):
    for i, axis in enumerate(fig_axes):
        ixs = np.hstack([o[i][0][0] for o in outliers if o[i][0][0].shape[0]])
        vls = np.hstack([o[i][0][1] for o in outliers if o[i][0][1].shape[0]])
        ixs = ixs.astype(int)
        axis.plot(time[ixs], vls, color)

def plot_check_quality(fingers, int1, int2, outliers, timestmp, axes, fig, 
                       subj, subj_diag):
    plot_per_axis(int1, timestmp, 'b', axes)
    plot_per_axis(int2, timestmp, 'g', axes)
    plot_per_axis_outlier(outliers, timestmp, 'k*', axes)
    axes[0].set_ylim(axes[0].get_ylim())
    axes[1].set_ylim(axes[1].get_ylim())
    plot_per_axis(fingers, timestmp, 'r', axes, **{'alpha':.1})
    axes[0].set_title(f'Diag:{subj_diag}| {subj}: doing Quality Inspection')
    axes[1].set_xlabel(f"Instructions. Prediction quality good for analysis?"
                       f"\n Press:  0) if no  1) if yes")
    fig.tight_layout()
    vals = [0, 1]
    res = get_key_resp(fig, vals)
    res = resp_corr_fig_bkg(fig, res)
    return res

def resp_corr_fig_bkg(fig, res):
    if res == 0 or res == 'bad':
        fig.patch.set_facecolor('xkcd:light red')
        res = 'bad'
    elif res == 1 or res == 'good':
        fig.patch.set_facecolor('xkcd:mint green')
        res = 'good'
    return res

def plot_get_times(fingers, int2,timestmp, axes, fig, subj, subj_diag):

    lines0 = axes[0].plot(timestmp, int2[:,0,:].T, '-', lw=2, picker=3)
    lines1 = axes[1].plot(timestmp, int2[:,1,:].T, '-', lw=2, picker=3)

    _ = axes[0].plot(timestmp, int2[:,0,:].T, 'k.', markersize=1)
    _ = axes[1].plot(timestmp, int2[:,1,:].T, 'k.', markersize=1)

    ln_clr= ['r', 'g']  
    n_side = len(lines0)/2
    n_r = np.arange(n_side, dtype='int')
    n_l = np.arange(n_side, n_side*2, dtype='int')

    for s, c in zip([n_r, n_l], ln_clr): # side and color
        for l in [ np.array(lines0)[s],  np.array(lines1)[s]]: # lines
            for j in l:
                j.set_color(c)

    plot_per_axis(fingers, timestmp, 'r', axes, **{'alpha':.1})
    axes[0].set_title(f'Diag:{subj_diag}| {subj}: doing R-L Beg-End times')
    axes[1].set_xlabel(f"Instructions. Select R/L beg/end."
                       " Click: Right (RED) beginning - end, Left (Green) beg."
                       f" - end time points. \n Press: "
                       "Escape) to reselect point. Enter) if good time point")
    fig.tight_layout()

    res = get_key_times(fig, axes)
    times = np.hstack([t[0] for t in res['inxs']])
    times_leng = np.hstack([timestmp[times[i + 1]] - timestmp[times[i]]
                            for i in [0,2]])
    return times, times_leng


def plot_frame_outlier(frame, coords_ori, coords_out, axis):
    axis.imshow(frame)
    mrkr = ['kx', 'r+']
    for i, vers in enumerate([coords_ori, coords_out]):
        for coor in vers:
            axis.plot(coor[0], coor[1], mrkr[i])

def plot_contig_frames(res, frame_num, fingers, int2, path_s, subj, relab):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    fram_n = frame_num - 1 if res == "p" else frame_num +1
    while True:
        fram = uio.get_prediction_frame(path_s, subj, fram_n)
        plot_frame_outlier(fram, fingers[:, :, fram_n],
                           int2[:, :, fram_n], ax)
        ax.set_title(f"Frame n: {fram_n}, outlier frame {frame_num}")
        ax.set_xlabel(f"Instructions. p) previous n) next"
                   " frame, r) Relabel preds, escape) finish viewing")
        res2 = get_key_resp(fig, vals=["escape", "p", "n", "r"])
        if res2 == 'escape':
            plt.close(fig)
            print("out loop")
            break
        elif res2 in ["p", "n"]:
            fram_n = fram_n - 1 if res2 == "p" else fram_n +1
            ax.clear()
        elif res2 in ["r"]:
            fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
            pred = plot_make_new_pred(fram, int2[:, :, fram_n], fig2, ax2, fram_n)
            relab.extend(pred)
            plt.close(fig2)
            for p in pred:
                int2[int(p[1]), :, int(p[0])] = p[2:]

    return relab



def plot_oulier_qc(outliers, fingers, int2, path_s, subj, axis, fig, relab):
    # reshape outliers, current very bad

    out_new = []
    for nf, f in enumerate(outliers):
        for ns, xy in enumerate(f):
            for xy_p, xy_v in zip(xy[0][0], xy[0][1]):
                out_new.append([nf, ns, int(xy_p), round(xy_v)])

    # Ensure same prediction diff axis (x, y) are not inspected twice
    frames_nums = [ o[2] for o in out_new]
    u_val, u_x, u_inv, u_c = np.unique(frames_nums, return_index=True, 
                                       return_inverse=True, return_counts=True)
    o_done = np.zeros(u_c.size, dtype=bool)

    for i, out in enumerate(out_new):
        frame_num = out[2]

        # Check if inspected previous axis (x, y)
        if o_done[u_inv[i]]:
            prev = frames_nums.index(frame_num)
            # Add resp from previous axis
            out_new[i].append(out_new[prev][-1])
            continue

        frame = uio.get_prediction_frame(path_s, subj, frame_num)
        coords_ori = fingers[:, :, frame_num]
        coords_out = int2[:, :, frame_num]
        while True:
            plot_frame_outlier(frame, coords_ori, coords_out, axis)
            axis.set_xlabel(f"Instructions. Outlier (red +) improved prediction?"
                           " Press:  0) if no  1) if yes, r) relabel, p - q) " 
                           " contiguous frames \n Then enter) next"
                           " escape) respond again")
            fig.tight_layout()
            # Select if good outlier
            res = get_key_resp(fig, vals=[0, 1, "p", "n", "r"])
            # Plot contiguous frames
            if res in ["p", "n"]:
                relab = plot_contig_frames(res, frame_num, fingers, int2,
                                           path_s, subj, relab)
                res = get_key_resp(fig, vals=[0, 1])
            elif res == "r":
                res = 0

                fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
                pred = plot_make_new_pred(frame, coords_out, fig2, ax2,
                                          frame_num)
                relab.extend(pred)
                plt.close(fig2)
                for p in pred:
                    coords_out[int(p[1]), :] = p[2:]
                    int2[int(p[1]), :, int(p[0])] = p[2:]
                plot_frame_outlier(frame, coords_ori, coords_out, ax2)
                axis.set_xlabel(f"Instructions. enter) to confirm"
                           " escape) respond again")


            res = resp_corr_fig_bkg(fig, res)

            out_val = int2[out[0], :, frame_num]
            axis.plot(out_val[0], out_val[1], "ro", markerfacecolor="none")
            # Accept answer
            out_insp = get_key_resp(fig, vals=['enter', 'escape'])
            if out_insp == 'enter':
                break
            elif out_insp == 'escape':
                axis.clear()
        axis.clear()
        fig.patch.set_facecolor('w')
        out_new[i].append(res)
        o_done[u_inv[i]] = True
    return out_new, relab


def plot_ts_inspection(out_checked, timestmp, int1, int2, path_s, subj,
                       subj_diag, axes, fig, ttl=None):
    axes[0].clear()
    axes[1].clear()
    if ttl:
        axes[0].set_xlabel(ttl, fontweight='bold')

    def plot_ts(int2):
        lines0 = axes[0].plot(timestmp, int2[:,0,:].T, '-', lw=1, picker=3)
        lines1 = axes[1].plot(timestmp, int2[:,1,:].T, '-', lw=1, picker=3)

        ln_clr= ['r', 'g']
        n_side = len(lines0)/2
        n_r = np.arange(n_side, dtype='int')
        n_l = np.arange(n_side, n_side*2, dtype='int')
    
        for s, c in zip([n_r, n_l], ln_clr): # side and color
            for l in [ np.array(lines0)[s],  np.array(lines1)[s]]: # lines
                for j in l:
                    j.set_color(c)
        return lines0, lines1

    lines0, lines1 = plot_ts(int2)
    # Plot outliers
    for out in out_checked:
        out[:4] = [int(i) for i in out[:4]]
        color = "go" if out[-1] == 'good' else "ro"
        axes[out[1]].plot(timestmp[out[2]], out[3], color,
                          markerfacecolor=None)

    plot_per_axis(int1, timestmp, 'b', axes, **{'alpha':.2})
    axes[0].set_title(f'Diag:{subj_diag}| {subj}: doing TS inspection')
    axes[1].set_xlabel(f"Instructions. Press Enter) to finish or click on a "
                       "time point for inspection. Then, press Enter) to plot "
                       "frame or Escape) to select another point")
    fig.tight_layout()
    new_pred, good_pred, relab = [], [], []
    while True:
        res, avline = get_clicked_times(fig, axes, 'k')
        if res['inxs']:
            frame_num, = res['inxs']
            pred, relab = plot_pred_relab(path_s, subj, frame_num, int1, int2,
                                          avline, [])
            if len(pred):
                new_pred.extend(pred)
                _ = [(l.remove(), l1.remove()) for l, l1 in zip(lines0,lines1)]
                for p in pred:
                    int2[int(p[1]),:,int(p[0])] = p[2:]
                lines0, lines1 = plot_ts(int2)
            else:
                good_pred.append(frame_num)
        else:
            break
    new_pred.extend(relab)
    return new_pred, good_pred


def plot_pred_relab(path_s, subj, frame_num, int1, int2, avline, relab, ttl=None):
    new_pred = []
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
    if ttl:
        ax2.set_title(ttl)
    frame = uio.get_prediction_frame(path_s, subj, frame_num)
    coords_ori = int1[:, :, frame_num]
    coords_out = int2[:, :, frame_num]
    while True:
        plot_frame_outlier(frame, coords_ori, coords_out, ax2)
        ax2.set_xlabel(f"Good prediction?"
                       f" Press:  0) if no  1) if yes, r) relabel, p - q) " 
                       " contiguous frames \n Then enter) next or"
                       " escape) respond again")
        ax2.set_xlabel(f"Good prediction?"
                       f" Press:  0) if no  1) if yes, p - q) " 
                       " contiguous frames \n Then enter) next or"
                       " escape) respond again")
        fig2.tight_layout()
        # Select if good outlier
        # res1 = get_key_resp(fig2, vals=[0, 1, "p", "n", "r"])
        res1 = get_key_resp(fig2, vals=[0, 1, "p", "n"])
        # Plot contiguous frames
        if res1 in ["p", "n"]:
            relab = plot_contig_frames(res1, frame_num, int1, int2, path_s,
                                       subj, relab)
            res1 = get_key_resp(fig2, vals=[0, 1])
        elif res1 == "r":
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            pred = plot_make_new_pred(frame, coords_out, fig, ax, frame_num)
            relab.extend(pred)
            plt.close(fig)
            for p in pred:
                coords_out[p[1], :] = p[2:]
            plot_frame_outlier(frame, coords_ori, coords_out, ax2)
            res1 = 0

        res1 = resp_corr_fig_bkg(fig2, res1)
        ax2.set_xlabel(f"{res1} prediction? Press: Enter) to proceed"
                       " or Esc) to respond again")

        # Confirm answer
        res_conf = get_key_resp(fig2, vals=['enter', 'escape'])
        if res_conf == 'enter':
            if res1 == 'good':
                plt.close(fig2)
                for l in avline:
                    l.set_color("g")
                break
            elif res1 == 'bad':
                pred = plot_make_new_pred(frame, coords_out, fig2, ax2, frame_num)
                new_pred.extend(pred)
                plt.close(fig2)
                for l in avline:
                    l.set_color("r")
                break
        elif res_conf == 'escape':
            ax2.clear()
            fig2.patch.set_facecolor('w')

    return new_pred, relab

def nan_inspec(nans_pred, path_s, subj, int1, int2, relab):
    good_pred = []
    bad_relab = []
    for frame_num in nans_pred:
        ttl = f"nan inspection, frame # {frame_num}"
        new_pred, relab = plot_pred_relab(path_s, subj, frame_num, int1, int2,
                                          [], relab, ttl)
        if len(new_pred):
            bad_relab.extend(new_pred)
        elif len(new_pred) == 0 :
            good_pred.append(frame_num)
    return good_pred, bad_relab, relab


def ret_new_pred(frame_num, coords, coords_out):
    coords, coords_out = np.squeeze(coords), np.squeeze(coords_out)
    pred = [np.hstack([frame_num, i, n]) for i, (n, o) in
                  enumerate(zip(coords, coords_out))
                  if not array_equal_nan(n, o)]
    pred = np.vstack(pred) if len(pred) else np.array([])
    return list(pred)


def array_equal_nan(a1, a2):
    """
    True if two arrays have the same shape and elements, False otherwise.
    1.19.0 Numpy
    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.
    Returns
    -------
    b : bool
        Returns True if the arrays are equal.
    """
    try:
        a1, a2 = np.asarray(a1), np.asarray(a2)
    except Exception:
        return False
    if a1.shape != a2.shape:
        return False
    # Handling NaN values
    a1nan, a2nan = np.isnan(a1), np.isnan(a2)
    # NaN's occur at different locations
    if not (a1nan == a2nan).all():
        return False
    # Shapes of a1, a2 and masks are guaranteed to be consistent by this point
    return bool(np.asarray(a1[~a1nan] == a2[~a1nan]).all())

def plot_make_new_pred(frame, coord, fig, ax, frame_num):
    def onclick1(event):
        resp['data'] = [event.mouseevent.xdata, event.mouseevent.ydata]

    def onkey(event):
        print(event.key)
        K = event.key
        resp['key'] = K
        try:
            resp['key']= int(resp['key'])
        except Exception: 
            pass

    def legend_coor(coord):
        legend = [s + f" - {c}" if np.isnan(c[0]) else s + f" - {c.astype(int)}" 
                  for s, c in zip(leg_names, coord)]
        return legend
    
    ax.clear()
    ax.imshow(frame, picker=True)
    leg_names = ["0 - r_inx ", "1 - r_thm", "2 - r_wrt ",
                 "3 - l_inx", "4 - l_thm", "5 - l_wrt "]
    legnd = legend_coor(coord)
    points = [ax.plot(c[0], c[1], 'x') for c in coord]
    ax.legend([c  for p in points for c in p], legnd)
    
    resp = {'data': [], 'inxs':[], 'key':None}
    new_coord = coord.copy()
    while True:
        cid2 = fig.canvas.mpl_connect('key_press_event', onkey)
        ax.set_xlabel('Press a number to change position, Enter) to finish')

        if resp['key'] == 'enter':
            break
        elif resp['key'] in  range(6):
            k = resp['key']
            resp['key'] = []
            crs, = ax.plot(new_coord[k][0], new_coord[k][1], 'ko',
                           markerfacecolor=None)
            ax.set_xlabel(f'Click on new position for "{leg_names[k]}", press '
                          "num again) for none, esc) to leave")
            while True:
                cid1 = fig.canvas.mpl_connect('pick_event', onclick1)
                if resp['key'] in ['enter', 'escape']:
                    break

                if len(resp['data']) or resp['key'] == k:
                    if resp['key'] == k:
                        x, y = np.nan, np.nan
                    else:
                        x, y = resp['data'][0], resp['data'][1]

                    if len(new_coord[k].shape)>1:
                        new_coord[k] = np.array([x, y])[:, np.newaxis]
                    elif len(new_coord[k].shape)==1:
                        new_coord[k] = np.array([x, y])

                    old_col = points[k][0].get_color()
                    new_p = ax.plot(x, y, color=old_col, marker="x",
                           markerfacecolor=None)

                    points[k][0].remove()
                    points.pop(k)
                    points.insert(k, new_p)
                    new_leg = legend_coor(new_coord)
                    ax.legend( [c  for p in points for c in p], new_leg)

                    fig.canvas.mpl_disconnect(cid1)
                    crs.remove()
                    resp = {'data': [], 'inxs':[], 'key':None}
                    break
                plt.pause(.1)
        plt.pause(.5)

    fig.canvas.mpl_disconnect(cid2)
    return ret_new_pred(frame_num, new_coord, coord)


# Events functions

def get_key_times(fig, axes):
    clrs = ['r', 'b', 'g', 'k']
    resp = {'data': [], 'inxs':[]}
    ins = ["R beg", "R end", "L beg", "L end"]
    for i, clr in enumerate(clrs):
        axes[1].set_xlabel(f"Instructions. Select {ins[i]}. \n Press: Escape) "
                           "to reselect point. Enter) if good time point")
        r, _ = get_clicked_times(fig, axes, clr)
        resp['inxs'].append(r['inxs'])
        resp['data'].append(r['data'])
    return resp


def get_clicked_times(fig, axes, clr):
    def onclick(event):
        ind = event.ind
        if len(ind) > 1:
            ind = ind[round(len(ind)/2)]
        x, y = event.artist.get_xdata()[ind], event.artist.get_ydata()[ind]
        resp['inxs']= [ind]
        resp['data'] = (x, y)

    def onkey(event):
        print(event.key)
        K = event.key
        resp['key'] = K

    resp = {'data': [], 'inxs':[], 'key':None}
    ax_vls = []
    previous_resp = []
    while True:
        cid1 = fig.canvas.mpl_connect('pick_event', onclick)
        cid2 = fig.canvas.mpl_connect('key_press_event', onkey)

        if resp['key'] == 'escape':
            resp = { 'data': [], 'inxs':[], 'key':None}
            for l in ax_vls:
                l.remove()
            ax_vls = []
        elif resp['key'] == 'enter':
            break

        if len(resp['inxs']) and clr and not resp['inxs'] == previous_resp:
            if ax_vls:
                for i, l in enumerate(ax_vls):
                    l.remove()
                ax_vls = []

            for ax in axes:
                ax_vls.append(ax.axvline(resp['data'][0], c=clr, lw=1))
            previous_resp = resp['inxs']
        plt.pause(.5)
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    return resp, ax_vls




def get_key_resp(fig, vals):
    "Returns key press in fig from``vals`` list"
    def onkey(event):
        print(event.key)
        K = event.key
        resp['key'] = K

    resp = {"responded":False, 'key':-1}
    while not resp["responded"]:
        cid = fig.canvas.mpl_connect('key_press_event', onkey)
        plt.pause(.2)
        try:
            resp['key']= int(resp['key'])
        except Exception: 
            pass
        if resp['key'] in vals:
            resp["responded"] = True
            fig.canvas.mpl_disconnect(cid)
    return resp['key']

