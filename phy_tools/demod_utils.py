# -*- coding: utf-8 -*-
"""
Created on Fri March 1 12:28:03 2019

@author: phil
"""

import numpy as np
import scipy as sp
import scipy.signal as signal
import ipdb
import copy
import time

import phy_tools.fp_utils as fp_utils

from phy_tools.plt_utils import plot_time_helper, waterfall_spec, plot_waterfall, ret_omega, gen_psd
from phy_tools.fil_utils import mov_avg, LPFilter, gen_fixed_poly_filter, fp_fil_repr
from phy_tools.fp_utils import comp_frac_width, sfi, ufi

from phy_tools.gen_utils import make_rrc_filter, hyst_trigger, use_log_tables, use_exp_tables, make_log_tables
from phy_tools.gen_utils import make_exp_tables, complex_rot

from numba import njit, float64, int64, complex128

from numpy.random import uniform
from scipy.stats import multivariate_normal as mvn

from sklearn.cluster import DBSCAN

from subprocess import check_output, CalledProcessError, DEVNULL
try:
    __version__ = check_output('git log -1 --pretty=format:%cd --date=format:%Y.%m.%d'.split(), stderr=DEVNULL).decode()
except CalledProcessError:
    from datetime import date
    today = date.today()
    __version__ = today.strftime("%Y.%m.%d")

channel_dict = dict.fromkeys(['start_bin', 'end_bin', 'start_slice', 'end_slice'])

def gen_h_est(in_vec, train_symbols, num_coefs=20):
    """
        H Matrix estimation technique.  Does not assume delta correlated training sequence.
    """
    r_vec = train_symbols[num_coefs:] #+ [0] * num_coefs  new_values
    x_vec = train_symbols[1:num_coefs+1][::-1] # old values

    X_T = np.matrix(sp.linalg.toeplitz(x_vec, r_vec))
    X = X_T.T
    h_est = np.linalg.inv(X.H * X)* X.H * np.matrix(in_vec).T
    return h_est


def lms(in_vec, train_symbols, num_taps=5, mu=.005, init_w=None, train_iters=1, train_offset=0):
    """
        Least Mean Squares equalizer.

        Uses the slicer object to generate distance metrics

        ==========
        Parameters
        ==========

            * in_vec : ndarray
                Input vector of complex symbols.
            * train_symbols : ndarray
                vector of training symbols
            * num_taps : int
                number of taps in adaptive filter.

        =======
        Returns
        =======

            * sym_out : ndarray
                vector of equalized symbols

            * wopt : ndarray
                vector of optimized adaptive filter taps.
    """
    # populate initial p vector -- cross-correlation vector and
    # R Matrix -- autocorrelation matrix.
    if init_w is None:
        w_taps = np.zeros((num_taps,), np.complex)
        # make first tap 1.
        w_taps[0] = 1 + 0j
    else:
        w_taps = copy.deepcopy(init_w)
        num_taps = len(w_taps)

    ret_vals = []
    # update taps 10 times on same data.
    for ii in range(train_iters):
        shift_reg = np.zeros((num_taps,), np.complex)
        for jj, (sym, train_val) in enumerate(zip(in_vec[train_offset:], train_symbols)):
            # taps_list.append(w_taps)
            # run transversal filter.
            shift_reg[1:] = shift_reg[:-1]
            shift_reg[0] = sym
            y_val = np.dot(np.conj(w_taps), shift_reg)
            error = train_val - y_val
            scale = np.dot(np.conj(shift_reg), shift_reg)
            w_incr = mu * np.conj(error) * shift_reg / scale
            w_taps = w_taps + w_incr
            # print('w_taps = {}, jj = {}'.format(w_taps, jj))

    shift_reg = np.zeros((num_taps,), np.complex)
    for val in in_vec:
        shift_reg[1:] = shift_reg[:-1]
        shift_reg[0] = val
        y_val = np.dot(np.conj(w_taps), shift_reg)
        ret_vals.append(y_val)

    return ret_vals, w_taps


def trig_setup(input_sig, offset, on_trigger, off_trigger):
    input_sig = input_sig[offset:-offset]
    pad_start = np.array([input_sig[0]] * offset)
    pad_end = np.array([input_sig[-1]] * offset)
    input_sig = np.concatenate((pad_start, input_sig, pad_end))
    on_vals = np.roll(input_sig + on_trigger, offset)
    off_vals = np.roll(input_sig + off_trigger, -offset)

    return input_sig, on_vals, off_vals


class EM_Algo_Gaussian(object):
    """
        ## Full Algorithm

        Initialize:

        1. Set number of clusters, $n$.
        2. Generate $n$ Gaussians with random initialized $\mu$ and $\Sigma$.  These are the $p_{X|Y}(x_i | y_j)$ distributions.

        E - Step (Expectation Step)

        1.  Compute marginals, $p_{X}(x_i)$ for all inputs, $x$.
        2.  Compute cluster conditionals, $p_{Y|X}(y_j|x_i)$, using (3)
        3.  Compute Cluster Likelihood score, $L=\displaystyle\sum\limits_{j}\log \left( p_{X}(x_i) \right)$

        M - Step (Maximization Step)

        1. Compute new cluster weights, $p_{Y}(y_j)$
        2. Compute new distribution means and variances :  $\mu_{Y}(y_j)$ and $\Sigma_{Y}(y_j)$

        Iterate until Likelihood score converges.
    """
    @staticmethod
    def comp_marginals(samps, weights, rvs):
        """
            Compute marginal probabilities (posteriors) given cluster distributions and cluster weightings/priors
            
            Args:
                samps : samples or measurements.
                weights : probability of class, j, likelihood of inclusion.   
                rvs : random variables.  Current Gaussian distribution of class j.
                
            Returns:
                marginal probabilities.

                Compute marginals by 

                $$
                \begin{aligned}
                    p_X(x_i) &= \displaystyle\sum\limits_{j}p(x_i, y_j)\\
                    &= \displaystyle\sum\limits_{j}p_{X|Y}(x_i | y_j)p_Y(y_j)
                \end{aligned}\tag{1}
                $$

                Where $p_Y(y_j)$ is the probability of class $j$.Using the Gaussian assumption the conditional probability (multivariate case), $p_{X|Y}(x_i | y_j)$, is given as:

                $$
                    p_{X|Y}(x_i | y_j) = \frac{1}{\left(2\pi\right)^{d/2}} |\Sigma|^{-1/2} \exp \left\{-\frac{1}{2} (x - \mu) \Sigma^{-1} (x - \mu)^T  \right\} \tag{2}
                $$
                
        """
        marg_probs = np.zeros(len(samps))
        probs = np.zeros((len(weights), len(samps)))
        for i, (w_i, rv) in enumerate(zip(weights, rvs)):
            probs[i] = w_i * rv.pdf(samps)

        marg_probs = np.sum(probs, axis=0)
        return marg_probs


    @staticmethod
    def comp_clus_cond(samps, marginals, weights, rvs):
        """
            Computes cluster Posterior probability given prior, W, and Likelihood (P(x|C) / P(x))
            Need to update conditional probability of class, $y$, given sample, $x$. Notice that the denominator is simply the marginal, $p(x_i)$ that is already given.

            $$
            \begin{aligned}
                p_{Y|X}(y_j|x_i) &= \frac{p_{X|Y}(x_i|y_j)p_{Y}(y_j)}{\displaystyle\sum\limits_{j}p_{X|Y}(x_i|y_j)p_{Y}(y_j))}\\
                &= \frac{p_{X|Y}(x_i|y_j)p_{Y}(y_j)}{p_{X}(x_i)}
            \end{aligned}\tag{3}
            $$
        """
        nclusters = len(weights)
        cond_probs = np.zeros((nclusters, len(samps)))
        for i in range(nclusters):
            cond_probs[i, :] = weights[i] * rvs[i].pdf(samps) / marginals

        return cond_probs


    @staticmethod
    def comp_new_weights(cond_probs):
        """
            Compute new cluster probabilities (P(C) = W_i).
            We use the conditional expectation mean to estimate cluster marginals or new priors.  
            Note that this is an estimate, not and identity.

            $$
                p_{Y}(y_j) \approx E(Y|X=x) = \frac{1}{n}\displaystyle\sum\limits_{i}p_{Y|X}(y_j|x_i) \tag{4}
            $$
        """
        return [np.sum(cluster) / len(cluster) for cluster in cond_probs]

    @staticmethod
    def comp_new_means(samps, cond_probs):
        """
            Compute new cluster means

            Estimate new distribution parameters

            Use probability mass function to sample mean and sample variance to compute new cluster distributions, $p_{X|Y}(x_i | y_j)$.

            $$
                \mu_{Y}(y_j) = \frac{1}{n}\displaystyle\sum\limits_{i}p_{X|Y}(x_i|y_i) \tag{5}
            $$

            $$
                \Sigma_{Y}(y_j) = \frac{1}{n}\displaystyle\sum\limits_{i}\left(x_i - \mu_{Y}(y_j)\right)\left(x_i - \mu_{Y}(y_j)\right)^T p_{X|Y}(x_i|y_i)\tag{6}
            $$
        """
        samps_dims = np.shape(np.atleast_2d(samps))
        means = []
        for cond_prob in cond_probs:
            if samps_dims[0] == 1:
                means.append(np.dot(samps, cond_prob) /  np.sum(cond_prob))
            else:
                means.append(np.matmul(np.atleast_2d(samps).T, np.atleast_2d(cond_prob).T).flatten() /  np.sum(cond_prob))

        return means

    @staticmethod
    def comp_new_sigmas(means, samps, cond_probs):
        """
            Compute new sigma values : 
        """
        sigma = []
        for mu, cond_prob in zip(means, cond_probs):
            diff_terms = samps - mu
            diff_terms = [np.atleast_2d(value) for value in diff_terms]
            num = np.sum([prob * np.matmul(dterm.T, dterm) for prob, dterm in zip(cond_prob, diff_terms)], axis=0)
            sigma.append(num / np.sum(cond_prob))

        return sigma


    @staticmethod
    def em_algo(samps, nclusters=2, mean_bnds=(-1, 1), var_bnd=1., eps=.001, num_dims=2):
        """
            Initialize:

            1. Set number of clusters, $n$.
            2. Generate $n$ Gaussians with random initialized $\mu$ and $\Sigma$.  These are the $p_{X|Y}(x_i | y_j)$ distributions.

            E - Step (Expectation Step)

            1.  Compute marginals, $p_{X}(x_i)$ for all inputs, $x$.
            2.  Compute cluster conditionals, $p_{Y|X}(y_j|x_i)$, using (3)
            3.  Compute Cluster Likelihood score, $L=\displaystyle\sum\limits_{j}\log \left( p_{X}(x_i) \right)$

            M - Step (Maximization Step)

            1. Compute new cluster weights, $p_{Y}(y_j)$
            2. Compute new distribution means and variances :  $\mu_{Y}(y_j)$ and $\Sigma_{Y}(y_j)$

            Iterate until Likelihood score converges.

        """
        # initial nclusters of gaussian randoms.
        rvs = []
        for i in range(nclusters):
            mean = uniform(mean_bnds[0], mean_bnds[1], num_dims)
            sigma = uniform(0, var_bnd, (num_dims, num_dims))
            sigma += np.eye(num_dims) * np.max(sigma)
            rvs.append(mvn(mean, sigma))

        # start with equal cluster weights
        weights = [1 / nclusters] * nclusters
        e_diff = 1_000_0000
        old_e = e_diff
        ret_rvs = [rvs]

        while e_diff > eps:
            marginals = EM_Algo_Gaussian.comp_marginals(samps, weights, rvs)
            cond_probs = EM_Algo_Gaussian.comp_clus_cond(samps, marginals, weights, rvs)
            new_e = np.sum(np.log(marginals))

            e_diff = np.abs(new_e - old_e)
            print(e_diff, new_e,)

            weights = EM_Algo_Gaussian.comp_new_weights(cond_probs)
            means = EM_Algo_Gaussian.comp_new_means(samps, cond_probs)
            sigmas = EM_Algo_Gaussian.comp_new_sigmas(means, samps, cond_probs)

            # create new rvs
            rvs = [mvn(mean, sigma) for (mean, sigma) in zip(means, sigmas)]

            ret_rvs.append(rvs)
            old_e = new_e

        return ret_rvs



def schmidl_cox_est(input_sig, block_size=256, corr_size=256, num_blocks=2, min_pwr=None):
    '''
    ==========
    Parameters
    ==========

        * input_sig : (ndarray) complex
            Input signal
        * corr_size : (int)
            number of samples that repeat in the preamble.
        * block_size: (int)
            number of samples between self-correlations.
        * num_blocks : (int)
            total number of correlated blocks in preamble per correlation.
        * min_pwr : (float)
            minimum power of auto_correlation values to be used for
            normalization.
        * numCorrs : int
            Total number of self correlations.

    =======
    Returns
    =======

        out : dict -- (ttiming_est_norm,phase_angle,r_d)

            * timing_est_norm : normalized timing estimate
                -- normalized by auto_correlation or power estimate.
            * phase_angle    : frequency offset estimate.
            * r_d      : auto_correlation estimate.

    '''
    sig_offsets = []
    # create offset signal vector for cross correlation
    for ii in range(num_blocks):
        offset = ii * block_size
        temp = np.roll(input_sig, -offset)
        sig_offsets.append(temp)

    # perform complex cross correlation (complex conjugate second term)
    corr_func = 0
    comp_conj = 0
    # comp_conj = 0
    for ii in range(num_blocks - 1):
        corr_func += np.conj(sig_offsets[ii]) * sig_offsets[ii + 1]
        comp_conj += np.abs(sig_offsets[ii + 1]) ** 2.

    # now normalize over block lengths.
    fil_b = np.ones((corr_size,)) / corr_size
    max_shift = (num_blocks - 2) * block_size

    p_d = signal.upfirdn(fil_b, corr_func)
    phase_angle = np.angle(p_d)
    # need to roll phase_angle by maximum shift
    phase_angle = np.roll(phase_angle, max_shift)
    r_d = signal.upfirdn(fil_b, comp_conj)

    idx = (r_d == 0.)
    print(any(idx))
    r_d[idx] = np.mean(r_d)
    if min_pwr is not None:
        idx = (r_d < min_pwr)
        r_d[idx] = min_pwr

    # r_d = np.roll(r_d, block_size // 4)
    timing_est_norm = (np.abs(p_d) ** 2) / (r_d ** 2)
    return (timing_est_norm, phase_angle, p_d, r_d)

def ret_fft_blocks(input_stream, fft_size, psd_input, overlap_segs, window):
    if psd_input:
        num_blocks = len(input_stream) // fft_size
        fft_blocks = np.reshape(input_stream[:num_blocks * fft_size], (num_blocks, fft_size))
    else:
        _, fft_blocks = waterfall_spec(input_stream, fft_size=fft_size, normalize=False, num_avgs=overlap_segs, window=window)

    return fft_blocks

class PeakPSDEnergyDetect(object):
    """
        Spectrogram segmentation based on peak detection.

        Note:
            This energy detector assumes input is already a 2-D Log scale spectrogram.  Each Spectral slice is passed 
            through a peak detector to find bins of interest.  Active bins are then grouped using a negative peak positive
            peak pattern.  Clustering (DBSCAN) is finally used to create separate active "channels" across frequency and
            time.

        Args:
            * on_trigger_db : active peaks are filtered by dB abover noise floor
            * cluster_eps : clustering expanding parameter.  % vector length in time/frequency plane.
            * min_samples : minimum number of samples for new cluster.
    """

    def __init__(self, fft_size=512, overlap_segs=4, window='blackmanharris',
                 k_vec=range(1, 4), on_trigger_db=6., cluster_eps=.05, min_samples=9, tweight=10., path='./'):

        self.fft_size = fft_size
        self.on_trigger_db = on_trigger_db
        self.peak_obj = PeakDetector(k_vec=k_vec)
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.overlap_segs = overlap_segs
        self.window = window
        self.w_vec = ret_omega(self.fft_size)[0]
        self.w_vec2 = ret_omega(self.fft_size * 2)[0]
        self.tweight = tweight
        self.chunk_size = self.fft_size * self.overlap_segs
        self.path = path

        # generate standard half-band filter for signal extraction
        self.hb_obj = LPFilter(half_band=True, num_taps=64)

    def gen_peaks(self, input_stream, on_trigger_db=6., psd_input=False):
        fft_blocks = ret_fft_blocks(input_stream, self.fft_size, psd_input, self.overlap_segs, self.window)
        mean_vals = np.mean(fft_blocks, axis=0)
        trig_vals = np.zeros_like(fft_blocks, dtype=bool)
        for ii, row in enumerate(fft_blocks):
            t1 = time.time()
            # look at average power for the bin across all fft_blocks
            pos_peaks = self.peak_obj.find_peaks(row)
            # thresh = on_trigger_db + np.median(row)
            pos_peaks = np.array([True if (value - mean_val >= on_trigger_db and value != 0.) else False for value, mean_val in zip(np.array(pos_peaks) * row, mean_vals)])
            pos_indices = pos_peaks.nonzero()[0]
            neg_indices = (self.peak_obj.find_peaks(-np.array(row))).nonzero()[0]

            # find first neg peaks on either side of pos peak
            # set trig values to be True from the left nearest negative peak to the right nearest negative peak relative
            # to the current positive peak.
            for pos_idx in pos_indices:
                temp = ((neg_indices - pos_idx) > 0).nonzero()[0]
                if len(temp):
                    pos_stop = neg_indices[temp[0]]
                else:
                    pos_stop = len(row)
                temp = ((pos_idx - neg_indices) > 0).nonzero()[0]
                if len(temp):
                    neg_stop = neg_indices[temp[-1]]
                else:
                    neg_stop = 0

                trig_vals[ii, neg_stop:pos_stop] = True

            if 0:
                t2 = time.time()
                format_str = ['-', '^', 'o']
                legend_str = ['Log Vals', 'Peaks', 'Chans']
                color = ['C{}'.format(i) for i in range(3)]
                peak_locs = pos_peaks * row
                peak_indices = [i for i, e in enumerate(pos_peaks) if e != 0.]
                chan_indices = [i for i, e in enumerate(trig_vals[ii, :]) if e == True]
                time_sig = [row, peak_locs[peak_indices], trig_vals[ii, chan_indices] * np.max(row)]
                markersize = [4.5] * len(time_sig)
                linewidth = [.9] * len(time_sig)
                w_vec = ret_omega(len(row))[0]
                w_vec_peaks = w_vec[peak_indices]
                w_vec_chans = w_vec[chan_indices]
                x_vec = [w_vec, w_vec_peaks, w_vec_chans]
                title = 'Peak Signals and Chans {}'.format(ii)  # title_fn)
                xlabel = r'$\sf{Frequency}$'
                # print("ii = {}, elapsed time = {}".format(ii, t2 - t1))
                plot_time_helper(time_sig, title=title, savefig=True, alpha=.8, linewidth=linewidth, plot_on=True,
                                 label=legend_str, format_str=format_str, x_vec=x_vec, xlabel=xlabel, legendsize=8,
                                 min_n_ticks=6, xprec=3, yprec=0, markersize=markersize, color=color,
                                 miny=np.min(row) - 5., maxy=np.max(row) + 10, minx=-1., maxx=1.)

                # plt.close('all')
        return trig_vals.T

    @staticmethod
    def gen_chans(yhat, x_indices, y_indices):
        """
            Uses clustering results, yhat, and performs reverse look-up with hot bin map, trig_vals,
            to create dictionary of channel "squares"
        """
        active_channels = []
        num_chans = np.max(yhat) + 1
        for chan in range(num_chans):
            indices = (yhat == chan).nonzero()[0]
            bin_start = np.min(y_indices[indices])
            bin_end = np.max(y_indices[indices])
            slice_start = np.min(x_indices[indices])
            slice_end = np.max(x_indices[indices]) + 1
            # print(bin_start, bin_end, slice_start, slice_end)
            active_channels = PeakPSDEnergyDetect.append_chan(bin_start, bin_end, slice_start, slice_end, active_channels)

        return active_channels

    @staticmethod
    def append_chan(bin_start, bin_end, slice_start, slice_end, active_channels):
        new_channel = copy.copy(channel_dict)
        new_channel['start_bin'] = bin_start
        new_channel['end_bin'] = bin_end
        new_channel['start_slice'] = slice_start
        new_channel['end_slice'] = slice_end
        active_channels.append(new_channel)

        return active_channels

    def process_input(self, iq_data, plot_on=False, savefig=False, psd_input=False, dpi=600, title=None, plot_psd=False,
                      pickle_fig=False, ms=30, plot_peaks=False, ret_neg_cfs=True, norm_slices=1000):
        t1 = time.time()
        trig_vals = self.gen_peaks(iq_data, self.on_trigger_db, psd_input)
        print("gen peaks time = {}".format(time.time() - t1))
        temp = trig_vals.T.nonzero()
        x_indices = temp[0]
        y_indices = temp[1]
        maxx = np.max(x_indices)
        maxy = np.max(y_indices)
        X = [[x/(self.tweight * maxx), y/norm_slices] for (x, y) in zip(x_indices, y_indices)]

        t1 = time.time()
        model = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples)
        yhat = model.fit_predict(X)
        print("cluster time = {}".format(time.time() - t1))

        # now enumerate clusters.
        active_channels = PeakPSDEnergyDetect.gen_chans(yhat, x_indices, y_indices)

        # filter out neg center frequency channels if ret_neg_cfs is False
        act_chans = []
        if not ret_neg_cfs:
            for chan in active_channels:
                curr_cf = self.w_vec2[chan['end_bin'] + chan['start_bin']]
                if np.sign(curr_cf) > 0.:
                    act_chans.append(chan)
        else:
            act_chans = copy.copy(active_channels)

        if title is None:
            title = r'$\sf{Final\ Output}$'

        if plot_on:
            fig_axes = plot_waterfall(iq_data, fft_size=self.fft_size, rotate_spec=True, window=self.window, title=title,
                                      plot_psd=plot_psd, normalize=True, num_avgs=self.overlap_segs, channels=active_channels,
                                      plot_on=plot_on, savefig=savefig, psd_input=psd_input, dpi=dpi, path=self.path)

            if plot_peaks:
                w_vec = ret_omega(self.fft_size)[0]
                freq_indices = [w_vec[idx] for idx in y_indices]
                # area = 2 * (1. / maxx * (1. / maxy))
                fig_axes[-1].scatter(x_indices, freq_indices, alpha=.05, c='C2', s=ms)

        return act_chans

    def extract_bursts(self, iq_data, plot_on=False, savefig=False, pickle_fig=False, dpi=600, title=None, time_offset=10_000,
                       baseband=True, resample=True, ret_neg_cfs=True, peak_detect=False, dec_rate=None, norm_slices=1000): 
        """ 
            Function performs basebanding, slicing, and resampling (via recursively applying half-band filter)
            to extract bursts.
        """
        active_channels = self.process_input(iq_data, plot_on=plot_on, savefig=savefig, psd_input=False,
                                             pickle_fig=pickle_fig, dpi=dpi, title=title, ret_neg_cfs=ret_neg_cfs,
                                             norm_slices=norm_slices)
        sig_list = []
        for channel in active_channels:
            curr_bw = (channel['end_bin'] - channel['start_bin']) / (self.fft_size / 2.)
            curr_cf = self.w_vec2[(channel['end_bin'] + channel['start_bin'])]
            num_hb_fils = int(np.log2(2. / (curr_bw + 1E-9))) if dec_rate is None else int(np.log2(dec_rate))
            start_idx = int(np.max((channel['start_slice'] * self.chunk_size  - time_offset, 0.)))
            end_idx = int(np.min((channel['end_slice'] * self.chunk_size + time_offset, len(iq_data)))) - 1
            # slice out time domain
            burst_data = iq_data[start_idx:end_idx]
            # tune to baseband
            # estimate pds peak as center frequency
            if peak_detect:
                omega, psd = gen_psd(burst_data)
                curr_cf = omega[np.argmax(psd)]

            if baseband:
                burst_data = complex_rot(burst_data, -curr_cf)
                if resample:
                    for i in range(num_hb_fils):
                        burst_data = self.hb_obj.hilbert_filter(burst_data)

            sig_list.append(burst_data)

        return active_channels, sig_list



class SpectrogramEnergyDetect(object):
    """
        Classical Energy based technique for Spectrogram segmentation.

        Note:
            This energy detector simply does an FFT of the input data stream.  Then each bin is run through a moving
            average and compared to the noise floor for thresholding of the "signal/no signal" decision.

        Args:
            * fft_size : FFT size applied to contiguous blocks of the input stream.
            * avg_len : CIC filter length applied to each bin of the FFT blocks.
            * on_trigger_db : dB offset to detect "turn on" of bin.
            * off_trigger_db : dB offset to detect "turn off" of bin.
            * overlap_segs : number of Spectrogram overlap segments.  Changes the Input/Output rate of the Spectrogram Estimate by [::overlap_segs]
            * avg_len : Moving average length of each bin of the Spectrogram estimate.
    """
    def __init__(self, fft_size=512, avg_len=8, on_trigger_db=6, off_trigger_db=6, overlap_segs=4, window='blackmanharris',
                 bin_allowance=10, slice_allowance=10, psd_input=False):

        self.fft_size = fft_size
        self.avg_len = avg_len
        # trigger values
        self.on_trigger_db = on_trigger_db
        self.off_trigger_db = off_trigger_db
        # PSD num averages
        self.overlap_segs = overlap_segs
        self.window = window

        self.bin_allowance = bin_allowance
        self.slice_allowance = slice_allowance

    def append_chan(self, curr_start, curr_end, slice_num, active_channels):
        new_channel = copy.copy(channel_dict)
        new_channel['start_bin'] = curr_start
        new_channel['end_bin'] = curr_end
        new_channel['start_slice'] = slice_num
        new_channel['end_slice'] = slice_num
        active_channels.append(new_channel)

        return active_channels

    def check_curr_chans(self, active_channels, curr_start, curr_end, slice_num):
        update = False

        if len(active_channels) == 0:
            return update
        for chan in active_channels:
            start_bool = abs(chan['start_bin'] - curr_start) < self.bin_allowance
            end_bool = abs(chan['end_bin'] - curr_end) < self.bin_allowance
            slice_bool = abs(chan['end_slice'] - slice_num) < self.slice_allowance
            # check if channel is inside existing channel
            start_inside = curr_start >= chan['start_bin']
            end_inside = curr_end <= chan['end_bin']
            # slice_inside = (slice_num <= chan['end_slice'])  #and ()

            if ((start_bool and end_bool) or (start_inside and end_inside)) and slice_bool:
                # widen current channel an update end_slice value
                chan['start_bin'] = np.min((chan['start_bin'], curr_start))
                chan['end_bin'] = np.max((chan['end_bin'], curr_end))
                chan['end_slice'] = slice_num
                update = True
                break

        return update

    def check_final_chans(self, active_channels, curr_chan):
        update = False
        curr_start = curr_chan['start_bin']
        curr_end = curr_chan['end_bin']
        curr_slice_start = curr_chan['start_slice']
        curr_slice_end = curr_chan['end_slice']
        if len(active_channels) == 0:
            return update
        for chan in active_channels:
            if chan == curr_chan:
                # skip identical channel
                continue
            start_bool = abs(chan['start_bin'] - curr_start) < self.bin_allowance
            end_bool = abs(chan['end_bin'] - curr_end) < self.bin_allowance
            start_slice_bool = abs(chan['start_slice'] - curr_slice_start) < self.slice_allowance
            end_slice_bool = abs(chan['end_slice'] - curr_slice_end) < self.slice_allowance
            adjacent_bool = (abs(chan['end_bin'] - curr_start) < self.bin_allowance) or (abs(chan['start_bin'] - curr_end) < self.bin_allowance)
            # overlap_bool = start_bool and end_bool and start_slice_bool and end_slice_bool
            adj_box_bool = adjacent_bool and start_slice_bool and end_slice_bool

            slice_seta = [*range(chan['start_slice'], chan['end_slice'] + 1)]
            slice_setb = [*range(curr_slice_start, curr_slice_end + 1)]
            slice_overlap = len(set(slice_seta) & set(slice_setb)) > 0

            bin_seta = [*range(chan['start_bin'], chan['end_bin'] + 1)]
            bin_setb = [*range(curr_start, curr_end + 1)]
            bin_overlap = len(set(bin_seta) & set(bin_setb)) > 0
            contain_bool = bin_overlap & slice_overlap

            # print(adj_box_bool, contain_bool)

            if adj_box_bool or contain_bool:
                # widen current channel an update end_slice value
                chan['start_bin'] = np.min((chan['start_bin'], curr_start))
                chan['end_bin'] = np.max((chan['end_bin'], curr_end))
                chan['end_slice'] = np.max((chan['end_slice'], curr_slice_end))
                chan['start_slice'] = np.min((chan['start_slice'], curr_slice_start))
                update = True
                break

        return update

    def gen_trig_frames(self, input_stream, plot_on=False, psd_input=False):
        """
            Method generates FFT frames, performs averaging, and dynamic thresholding to determine active bins.

            Args:
                * input_stream : IQ input stream
                * plot_on : Boolean used for debug plots.

            Returns:
                * trig_vals : Matrix of active bins (frequency is axis 0 and time axis 1.)
        """
        fft_blocks = ret_fft_blocks(input_stream, self.fft_size, psd_input, self.overlap_segs, self.window)
        fft_blocks = fft_blocks.T

        if plot_on:
            plot_waterfall(input_stream, fft_size=self.fft_size, rotate_spec=True, window=self.window, title='single_average_waterfall',
                           plot_psd=True, normalize=True, num_avgs=self.overlap_segs, plot_png=True, psd_input=psd_input)
            fft_block = fft_blocks[255, :]
            plot_time_helper(fft_block, title='sample block', savefig=True)

        trig_vals = np.zeros(np.shape(fft_blocks), dtype=bool)

        offset = self.avg_len // 2
        trig_offset = np.max((10, offset))
        for i, row in enumerate(fft_blocks):
            if self.avg_len > 1:
                log_vals = mov_avg(row, self.avg_len)[offset:-offset]
                pad_start = np.array([log_vals[0]] * offset)
                pad_end = np.array([log_vals[-1]] * offset)
                log_vals = np.concatenate((pad_start, log_vals, pad_end))
            else:
                log_vals = row
            on_vals = np.roll(log_vals + self.on_trigger_db, trig_offset)
            off_vals = np.roll(log_vals + self.off_trigger_db, -trig_offset)

            trig_temp = hyst_trigger(on_vals, off_vals, log_vals)
            trig_vals[i, :] = trig_temp

            if plot_on:
                title = 'Example FFT Slice {}'.format(i)
                plot_time_helper([on_vals, off_vals, log_vals, 40*np.array(trig_temp)], label = ['On Thresh', 'Off Thresh', 'Curr', 'Trigger'],
                                 title=title, plot_on=False, savefig=True)
                print("generated {}".format(title))
                # plt.close('all')
        return trig_vals

    def segment_blocks(self, trig_vals, active_channels=None, start_slice=0):
        """
            Method uses the output of gen_trig_frames to segment the spectrogram in to active time/frequency blocks

            Args:
                * trig_vals : Matrix of active bins (frequency is axis 0 and time axis 1.)
                * active_channels : current list of active channels
                * start_slice : current time offset (number of FFT slices).  Makes the code re-entrant when breaking up
                    large input streams into chunks of data.
        """
        # walk down each column
        trig_vals = np.array(trig_vals)
        if active_channels is None:
            active_channels = []

        fft_slices = trig_vals.T
        for slice_num, row in enumerate(fft_slices):
            slice_num += start_slice
            if any(np.asarray(row)):
                active_indices = row.nonzero()[0]  #np.where(np.asarray(row)==True)[0]
                curr_start = active_indices[0]
                # find next start.
                next_start = (np.diff(active_indices) >= 3).nonzero()[0] + 1
                if len(next_start) == 0:  # reached the end of new channels.
                    next_start = np.array([0])
                else:
                    next_start = np.concatenate(([0], next_start))

                for idx in range(len(next_start)):
                    curr_start = active_indices[next_start[idx]]
                    try:
                        curr_end = active_indices[next_start[idx + 1] - 1]
                    except:
                        curr_end = active_indices[-1]

                    # check current channels
                    update = self.check_curr_chans(active_channels, curr_start, curr_end, slice_num)
                    if update is False:
                        active_channels = self.append_chan(curr_start, curr_end, slice_num, active_channels)
            ipdb.set_trace()
        return active_channels, slice_num

    def filter_channels(self, input_stream, active_channels, plot_on=False, savefig=False, psd_input=False):
        """
            Currently filters active channels by length in time and if they overlap take the large one
        """
        final_channels = []
        # filter out short bursts
        for chan in active_channels:
            time_len = chan['end_slice'] - chan['start_slice']
            if time_len > 4:
                final_channels.append(chan)

        final_copy = copy.copy(final_channels)
        if len(final_copy) != 0:
            for chan in final_channels:
                update = self.check_final_chans(final_copy, chan)
                # if channels were combined then remove chan from channels list
                if update:
                    final_copy.remove(chan)

        if plot_on:
            plot_waterfall(input_stream, fft_size=self.fft_size, rotate_spec=True, window=self.window, title=r'$\sf{Final Output}$',
                           plot_psd=True, normalize=True, num_avgs=self.overlap_segs, channels=final_copy,
                           plot_on=plot_on, savefig=savefig, psd_input=psd_input)

        return final_copy

    def process_input(self, iq_data, plot_on=False, active_channels=None, start_slice=0, savefig=False, psd_input=False):

        trig_vals = self.gen_trig_frames(iq_data, plot_on=False, psd_input=psd_input)
        active_channels, last_slice = self.segment_blocks(trig_vals, active_channels, start_slice=start_slice)
        start_slice = last_slice + 1

        # print("# channels = {}".format(len(active_channels)))
        # testa = [print(chan) for chan in active_channels]
        active_channels = self.filter_channels(iq_data, active_channels, plot_on=plot_on, savefig=savefig, psd_input=psd_input)

        return active_channels, start_slice

class PeakDetector(object):
    """
        Uses peak detector described in "An Efficient Algorithm for Automatic Peak Detection in Noisy Periodic and Quasi-Periodic
        Signals" by Felix Scholkmann, Jens Boss and Martin Wolf.  The algorithm is named Automatic Multiscale-based Peak Detection
        (AMPD)

        There are four (4) components to the algorithm:
        1.  Computing Local Maxima Scologram (LMS) The LMS is calculated by assigned 0's to a Matrix M, where each row indicates
        a different scale, k.  Scale in this context is the number of samples pre-cursor and post-cursor.  If not a peak, M is
        populated with a random number between 1 and 2.  The original matrix is k (user selectable) rows by N columns.  Where N is
        the number of input samples.

        2.  Row-wise summation of the LMS matrix, M.  Determining largest scale, k, with the largest number of peaks.  The
        variable lambda indicates the largest scale with maximum peaks.

        3.  New matrix is generated with lambda rows and N columns.
        4.  The last step in the algorithm is to compute the column-wise standard deviation of the M matrix.  All indices where
        sigma equals 0 are stored away as the detected peaks.
    """
    def __init__(self, k_vec=[1, 2, 3, 4], alpha=1.):

        self.k_vec = np.array(k_vec, dtype=np.int64)
        self.alpha = alpha
        self.kmin = k_vec[0]

    @staticmethod
    @njit(float64[:,:](int64[:], float64[:], float64),cache=False)
    def comp_lms(k_vec, time_series, alpha):
        """
            Computes M matrix based on k vector and input
        """
        N = len(time_series)
        M = np.zeros((len(k_vec), N), dtype=np.float64)
        rand_values = np.random.uniform(0.0001, 1., size=np.shape(M)) + alpha
        for nn, k in enumerate(k_vec):
            end_offset = N - k
            for i in range(N):
                curr_val = time_series[i]
                prev_val = time_series[i-k]
                fut_val = time_series[(i+k) % N ]
                temp = rand_values[nn, i]
                if i >= k and i <= end_offset:
                    if curr_val >= prev_val and curr_val >= fut_val:
                        # if peak then write 0 to matrix
                        temp = 0.
                M[nn, i] = temp
        return M

    @staticmethod
    @njit(float64[:](float64[:,:], int64), cache=False)
    def comp_std(m_matrix, lbda):
        scale_fac = 1. / (lbda - 1)
        num_rows = np.shape(m_matrix)[0]
        std_dev = np.zeros((num_rows,), dtype=np.float64)
        for i in range(num_rows):  #row in enumerate(m_matrix):  # this should be size N
            row = m_matrix[i]
            sum_term = (1./lbda) * np.sum(row)
            diff_sum = 0.
            for val in row:
                diff_sum += (val - sum_term) ** 2.

            std_dev[i] = scale_fac * np.sqrt(diff_sum)

        return std_dev


    def rescale_lms(self, M, row_sum):
        lbda = np.argmin(row_sum) + 1  # paper numbers by 1 -> k.
        return M[:lbda][:], (lbda + self.kmin)

    def find_peaks(self, time_series):

        M = PeakDetector.comp_lms(self.k_vec, time_series.astype(np.float64), np.float64(self.alpha))

        row_sum = np.sum(M, 1)  #self.row_sum(M)
        M_mod, lbda = self.rescale_lms(M, row_sum)
        Mt = np.transpose(M_mod)
        std_dev = PeakDetector.comp_std(Mt, lbda)
        return (np.array(std_dev) == 0)

class AGCModule(object):
    """
        AGC Class:  Provides functions for performing Automatic Gain
        Control

        =======
        Methods
        =======

            * agc_fixed : Performs fixed point AGC simulation.
            * agc_float : Performs floating point AGC simulation.

        ==========
        Parameters
        ==========

            * track_range    : float
                The number of dB from the reference that the integrator
                uses the alpha_track multiplier.
            * alpha_track    : float
                alpha applied to integrator during tracking.
            * alpha_overflow : float
                alpha applied to integrator when system is in overflow.
            * alpha_acquire : float
                alpha applied to integrator when system is trying to
                acquire AGC lock.
    """
    def __init__(self, track_range=3., alpha_track=.004, alpha_overflow=.4, alpha_acquire=.04):

        self.track_range = track_range
        self.alpha_track = alpha_track
        self.alpha_overflow = alpha_overflow
        self.alpha_acquire = alpha_acquire

        self.logtrack_range = np.log(10.**(self.track_range / 10.))

    def agc_fixed(self, in_fi, ref_lin, port_a_bits=25, log_width=16, low_sig_bits=0):
        """
            Function performs AGC based on Log and EXP table lookups.
            Systems is a full functioning feedback AGC loop.

            ==========
            Parameters
            ==========

                * in_fi : sfi object
                    input vector can be complex or real.
                * ref_lin : float
                    reference voltage level in linear units.
                * port_a_bits : int
                    Maximum number of bits on an input to a multiplier
                    Typically 24 bits for Xilinx architectures.
                * log_width : int
                    Log table bit width
                * low_sig_bits : int
                    Lower limit of signal magnitude that causes the AGC loop
                    to suspend.
        """
        qvec = in_fi.qvec

        max_alpha = np.max((self.alpha_track, self.alpha_acquire, self.alpha_overflow))
        alpha_qvec = (port_a_bits, comp_frac_width(max_alpha, port_a_bits, signed=1))

        test_val = sfi(0, qvec)
        max_pos = test_val.range.max

        exp_width = in_fi.word_length

        input_abs = np.abs(in_fi.float)
        reg_low_sig = (2.**low_sig_bits - 1) * 2.**(-in_fi.frac_length)
        temp = sfi(ref_lin)

        alpha_track = np.floor(self.alpha_track * 2.**alpha_qvec[1]) * 2.**-alpha_qvec[1]
        alpha_overflow = np.floor(self.alpha_overflow * 2.**alpha_qvec[1]) * 2.**-alpha_qvec[1]
        alpha_acquire = np.floor(self.alpha_acquire * 2.**alpha_qvec[1]) * 2.**-alpha_qvec[1]

        in_tuple = (in_fi, log_width, exp_width)

        (log_combined_fi, exp_combined_fi, mult_value, log_qvec, exp_qvec) = self.gen_fix_tables(*in_tuple)

        # Create Log and EXP Tables
        log_ref = np.log(ref_lin)
        scale_fac = 2**(log_qvec[1])

        q_input = ufi(0, qvec, overflow='saturate')
        q_out = sfi(0, qvec, overflow='saturate')
        accum_fi = sfi(0, log_qvec, overflow='saturate')

        output = []
        gain_vec = []
        accum = 0
        gain = 1
        for (ii, (val_mag, val)) in enumerate(zip(input_abs, in_fi.float)):
            temp = val_mag * gain
            temp_q = q_input.ret_quant_float(temp)[0]

            output.append(q_out.ret_quant_float(val * gain)[0])

            log_val = use_log_tables(temp_q, log_combined_fi, log_qvec, qvec)
            #  this is already in fixed point representation.
            # ipdb.set_trace()
            ref_diff = log_ref - log_val
            alpha_int = alpha_track
            if val_mag <= reg_low_sig:
                alpha_int = 0
            elif temp_q >= max_pos:
                alpha_int = alpha_overflow
            elif abs(ref_diff) > self.logtrack_range:
                alpha_int = alpha_acquire

            # apply alpha_track factor.  This is in Log
            alpha_out = ref_diff * alpha_int
            accum += alpha_out
            accum_q = accum_fi.ret_quant_float(accum)[0]

            # exponential lookup to get linear value.
            # print("accum = {}".format(accum))
            gain = use_exp_tables(accum_q, exp_combined_fi, log_qvec, exp_qvec, mult_value)
            # print("gain = {}, actual value = {}".format(gain, np.exp(accum_q)))
            gain_vec.append(gain)

        return np.asarray(output), np.asarray(gain_vec)

    def gen_fix_tables(self, fi_obj, log_width=16, exp_width=16):
        """
            Generates Log and Exp LUT tables to be used in fixed point implementations of the Log and Exp conversion blocks

            ==========
            Parameters
            ==========

                * in_fi : sfi object
                    input vector can be complex or real.
                * ref_lin : float
                    reference voltage level in linear units.
                * port_a_bits : int
                    Maximum number of bits on an input to a multiplier
                    Typically 24 bits for Xilinx architectures.
                * log_width : int
                    Log table bit width
                * exp_width : int
                    Exp table bit width
                * log_sig_bits : int
                    Lower limit of signal magnitude that causes the AGC loop
                    to suspend.
        """
        word_length = fi_obj.word_length
        mag_qvec = (word_length, word_length - 1)

        # Create Log and EXP Tables
        (log_combined_fi, log_qvec) = make_log_tables(mag_qvec, log_width)
        (exp_combined_fi, exp_qvec, shift_qvec, mult_value) = make_exp_tables(log_qvec, exp_width, max_shift=word_length-1)

        return (log_combined_fi, exp_combined_fi, mult_value, log_qvec, exp_qvec)

    def agc_float(self, in_vec, ref_lin, low_sig_val=0, max_val=2000, sd_samps=100):
        """
            Function performs AGC based on Log and EXP table lookups.
            Systems is a full functioning feedback AGC loop.

            ==========
            Parameters
            ==========

                * in_vec : ndarray
                    input vector can be complex or real.
                * ref_lin : float
                    reference voltage level in linear units.
                * logSigVal : float
                    Lower limit of signal magnitude that causes the AGC loop
                    to suspend.
                * max_val : float
                    Upper signal value that indicates an AGC overflow.
                * sd_samps : shutdown samples.  Number of consecutive samples where agc_on is 0 that indicates the end
                    of a burst

        """
        output = []
        gain = 1

        log_ref = np.log(ref_lin)
        input_abs = np.abs(in_vec)
        gain_vec = []
        agc_on = []
        burst_ind = []
        curr_burst_ind = 0
        burst_cnt = sd_samps

        accum = 0

        for (ii, (val_mag, val)) in enumerate(zip(input_abs, in_vec)):
            temp = val_mag * gain
            alpha_int = self.alpha_track
            ref_diff = log_ref - np.log(temp + .0001)  # small factor to avoid NaNs
            if val_mag <= low_sig_val:
                alpha_int = 0
            elif temp >= max_val:
                alpha_int = self.alpha_overflow
            elif abs(ref_diff) > self.logtrack_range:
                alpha_int = self.alpha_acquire

            if val_mag <= low_sig_val:
                agc_on.append(0)
                burst_cnt -= 1
                if burst_cnt == 0:
                    curr_burst_ind = 0
            else:
                curr_burst_ind = 1
                burst_cnt = sd_samps
                agc_on.append(1)

            burst_ind.append(curr_burst_ind)
            alpha_out = ref_diff * alpha_int
            # perform integration.
            accum += alpha_out

            # slice value to perform table lookup.
            gain = np.exp(accum + .0001)
            gain_vec.append(gain)
            output.append(gain * val)

        return np.asarray(output), np.asarray(gain_vec), np.asarray(agc_on), np.asarray(burst_ind)

    def agc_float_simple(self, in_vec, ref_lin, low_sig_val=0, max_val=2000, sd_samps=100):
        """
            Function performs AGC based on Log and EXP table lookups.
            Systems is a full functioning feedback AGC loop.

            ==========
            Parameters
            ==========

                * in_vec : ndarray
                    input vector can be complex or real.
                * ref_lin : float
                    reference voltage level in linear units.
                * logSigVal : float
                    Lower limit of signal magnitude that causes the AGC loop
                    to suspend.
                * max_val : float
                    Upper signal value that indicates an AGC overflow.
                * sd_samps : shutdown samples.  Number of consecutive samples where agc_on is 0 that indicates the end
                    of a burst

        """
        # output = []
        gain_vec = []
        gain = 1
        gain_log = 0
        input_log = np.log(np.abs(in_vec) + .0001)
        log_ref = np.log(ref_lin)
        low_sig_log = np.log(low_sig_val + .0001)
        max_val_log = np.log(max_val)

        for log_mag in input_log:
            curr_log = log_mag + gain_log
            ref_diff = log_ref - curr_log
            alpha_int = self.alpha_track
            if log_mag <= low_sig_log:
                alpha_int = 0
            elif curr_log >= max_val_log:
                alpha_int = self.alpha_overflow
            elif abs(ref_diff) > self.logtrack_range:
                alpha_int = self.alpha_acquire

            alpha_out = ref_diff * alpha_int
            # perform integration.
            gain_log += alpha_out
            # slice value to perform table lookup.
            gain_vec.append(gain_log)

        ret_gain = np.exp(np.asarray(gain_vec) + .0001)
        output = ret_gain * in_vec
        return output, ret_gain

    @staticmethod
    @njit(complex128[:](complex128[:], float64, float64, float64, float64, float64, float64, float64),cache=False)
    def agc_loop(samples, ref_lin, alpha_track, alpha_overflow, alpha_acquire, logtrack_range,
                 low_sig_val=0., max_val=20.):
        """
            Function performs AGC based on Log and EXP table lookups.
            Systems is a full functioning feedback AGC loop.

            ==========
            Parameters
            ==========

                * in_vec : ndarray
                    input vector can be complex or real.
                * ref_lin : float
                    reference voltage level in linear units.
                * logSigVal : float
                    Lower limit of signal magnitude that causes the AGC loop
                    to suspend.
                * max_val : float
                    Upper (linear) signal value that indicates an AGC overflow.


        """
        output = np.zeros(len(samples), dtype=np.complex128)
        gain_log = 0  # zero out the accumulator.
        log_ref = np.log(ref_lin)
        low_sig_log = np.log(low_sig_val + .0001)
        max_val_log = np.log(max_val)

        # input_log = np.log(np.abs(samples) + .0001)
        for i, sample in enumerate(samples):
            log_mag = np.log(np.abs(sample) + .0001)
            curr_log = log_mag + gain_log
            ref_diff = log_ref - curr_log
            alpha_int = alpha_track
            if log_mag <= low_sig_log:
                alpha_int = 0
            elif curr_log >= max_val_log:
                alpha_int = alpha_overflow
            elif abs(ref_diff) > logtrack_range:
                alpha_int = alpha_acquire

            alpha_out = ref_diff * alpha_int
            # perform integration.
            gain_log += alpha_out
            # slice value to perform table lookup.
            # gain_vec[i] = gain_log
            # convert back to linear gain.
            lin_gain = np.exp(gain_log + .0001)
            output[i] = lin_gain * sample

        return output

    def __call__(self, samples, ref_lin, low_sig_val=0, max_val=20):

        samples = np.array(samples, dtype=np.complex128)
        alpha_track = np.float64(self.alpha_track)
        alpha_overflow = np.float64(self.alpha_overflow)
        alpha_acquire = np.float64(self.alpha_acquire)
        logtrack_range = np.float64(self.logtrack_range)
        low_sig_value = np.float64(low_sig_val)
        max_val = np.float64(max_val)

        return AGCModule.agc_loop(samples, np.float64(ref_lin), alpha_track, alpha_overflow,
                                  alpha_acquire, logtrack_range, low_sig_val, max_val)


class AGCModule_spdata(object):
    """
        AGC Class:  Provides functions for performing Automatic Gain
        Control

        =======
        Methods
        =======

            * agc_fixed : Performs fixed point AGC simulation.
            * agc_float : Performs floating point AGC simulation.

        ==========
        Parameters
        ==========

            * track_range    : float
                The number of dB from the reference that the integrator
                uses the alpha_track multiplier.
            * alpha_track    : float
                alpha applied to integrator during tracking.
            * alpha_overflow : float
                alpha applied to integrator when system is in overflow.
            * alpha_acquire : float
                alpha applied to integrator when system is trying to
                acquire AGC lock.
    """
    def __init__(self, track_range=3., alpha_track=.004, alpha_overflow=.4, alpha_acquire=.04):

        self.track_range = track_range
        self.alpha_track = alpha_track
        self.alpha_overflow = alpha_overflow
        self.alpha_acquire = alpha_acquire

        self.logtrack_range = np.log(10.**(self.track_range / 10.))

    @staticmethod
    @njit(complex128[:](complex128[:], float64, float64, float64, float64, float64, float64, float64),cache=False)
    def agc_loop(samples, ref_lin, alpha_track, alpha_overflow, alpha_acquire, logtrack_range,
                 low_sig_val=0., max_val=20.):
        """
            Function performs AGC based on Log and EXP table lookups.
            Systems is a full functioning feedback AGC loop.

            ==========
            Parameters
            ==========

                * in_vec : ndarray
                    input vector can be complex or real.
                * ref_lin : float
                    reference voltage level in linear units.
                * logSigVal : float
                    Lower limit of signal magnitude that causes the AGC loop
                    to suspend.
                * max_val : float
                    Upper (linear) signal value that indicates an AGC overflow.


        """
        output = np.zeros(len(samples), dtype=np.complex128)
        gain_log = 0  # zero out the accumulator.
        log_ref = np.log(ref_lin)
        low_sig_log = np.log(low_sig_val + .0001)
        max_val_log = np.log(max_val)

        # input_log = np.log(np.abs(samples) + .0001)
        for i, sample in enumerate(samples):
            log_mag = np.log(np.abs(sample) + .0001)
            curr_log = log_mag + gain_log
            ref_diff = log_ref - curr_log
            alpha_int = alpha_track
            if log_mag <= low_sig_log:
                alpha_int = 0
            elif curr_log >= max_val_log:
                alpha_int = alpha_overflow
            elif abs(ref_diff) > logtrack_range:
                alpha_int = alpha_acquire

            alpha_out = ref_diff * alpha_int
            # perform integration.
            gain_log += alpha_out
            # slice value to perform table lookup.
            # gain_vec[i] = gain_log
            # convert back to linear gain.
            lin_gain = np.exp(gain_log + .0001)
            output[i] = lin_gain * sample

        return output

    def __call__(self, samples, ref_lin, low_sig_val=0, max_val=20):

        samples = np.array(samples, dtype=np.complex128)
        alpha_track = np.float64(self.alpha_track)
        alpha_overflow = np.float64(self.alpha_overflow)
        alpha_acquire = np.float64(self.alpha_acquire)
        logtrack_range = np.float64(self.logtrack_range)
        low_sig_value = np.float64(low_sig_val)
        max_val = np.float64(max_val)

        return AGCModule_spdata.agc_loop(samples, np.float64(ref_lin), alpha_track, alpha_overflow,
                                         alpha_acquire, logtrack_range, low_sig_val, max_val)


def max_eye(comp_vec, num_offset, trig_sig, num_samps=64):
    """
        Returns the sample offset resulting in the maximum eye opening.
        This is found by 'trying' all possible sample offsets and accumulating
        the absolute values of the symbols.  The maximum value after a given
        number of samples is declared the winner and is used by follow on
        processing.

        ==========
        Parameters
        ==========

        comp_vec : np.complex -- 1d Array
            Complex vector to be processed

        num_offset : np.int -- 1d Array
            Number of sample offset to try, also indicates the
            downsampling rate.

        trig_sig : ndarray (np.bool)
            Boolean array that indicates the sample number to start the
            'voting' process.

        num_samps : np.int
            Number of samples to accumulate absolute value metrics.

        =======
        Returns
        =======

        new_trig_sig : ndarray (np.bool)
            Returns a phase adjusted trigger signal.

    """

    comp_vec_int = np.atleast_1d(comp_vec)
    trig_sig = np.atleast_1d(trig_sig)
    trig_indices = [i for i, val in enumerate(trig_sig) if val]

    trig_indices = trig_indices[0]

    new_trig_sig = np.zeros(np.shape(trig_sig), dtype=np.bool)

    for trig_idx in trig_indices:
        comp_vec_int = comp_vec_int[trig_idx:trig_idx + num_samps * num_offset]
        accum_vec = np.zeros((num_offset,))
        sqr_val = np.abs(comp_vec_int)
        for (ii, val) in enumerate(sqr_val):
            phase = ii % num_offset
            accum_vec[phase] += val

        win_phase = np.argmax(accum_vec)
        if win_phase >= num_offset // 2:
            win_phase = win_phase - num_offset
        new_trig_sig[win_phase + trig_idx] = True

    return new_trig_sig


def gain_calc_kpki(loop_eta, loop_bw_ratio):
    """
        Function compute the kp and ki values for the loop filter of the
        synchronization system.
    """

    # PLL parameters
    K_pd = 1  # abs(err_gain); %error detector gain
    Kv = 1  # sqrt(EsAvg); %2*pi/4;
    Fn_Fs = loop_bw_ratio  # sampsBaud;
    theta = Fn_Fs  # *pi; %(Fn/Fs)*pi;
    eta = loop_eta
    ki_s = (4. * theta**2.) / (1 + 2. * eta * theta + theta**2.)
    kp_s = (4. * eta * theta) / (1 + 2 * eta * theta + theta**2.)
    kp = kp_s / (Kv * K_pd)
    ki = ki_s / (Kv * K_pd)

    return (kp, ki)

def gain_calc_walls(loop_eta, loop_bw, kpd):
    """
        Function compute the kp and ki values for the loop filter of the
        synchronization system.  This function uses the calculations outlined by Andy Walls.
    """

    omega_nt = loop_bw / np.pi
    omega_dt = omega_nt * np.sqrt(1 - loop_eta ** 2.)
    kpd_term = 2. / kpd
    exp_term = np.exp(-loop_eta * omega_nt)
    sinh_term = np.sinh(loop_eta * omega_nt)
    cos_term = np.cos(omega_dt)
    cosh_term = np.cos(omega_dt)
    kp = kpd_term * np.exp(-loop_eta * omega_nt) * sinh_term
    if loop_eta < 1.:
        ki = kpd_term * (1 - exp_term * (sinh_term + cos_term))
    elif loop_eta == 1.:
        ki = kpd_term * (1 - exp_term * (sinh_term + 1))
    else:
        ki = kpd_term * (1 - exp_term * (sinh_term + cosh_term))

    return kp, ki


class PolyTimingRec(object):
    def __init__(self, samps_baud_in=2, samps_baud_out=1, P=32, beta=.2, mf=None, taps_per_path=12):
        """
            Synchronization loop based on Fred Harris design specified "Multirate
            Digital Filters for Symbol Timing Synchronization in Software
            Defined Radios” Specifically Fig 8 of the paper."

            ==========
            Parameters
            ==========

                * samps_baud_in : float
                    nominal or estimated samples per baud at the input to the
                    synchronizer.
                * samps_baud_out : int
                    integer number of desired samples per baud at the output.
                * P            : int
                    number of polyphase filter paths.
                * alpha        : float (None)
                    RRC alpha factor -- used when d
                * mf            : ndarray (float)
                    Matched filter coefficients.  If supplied alpha variable is
                    ignored.

            =======
            Returns
            =======

                 Attributes:
                     * mf : ndarray
                         Matched Filter
                     * dmf : ndarray
                         differential Matched Filter.
                     * poly_fil : ndarray
                         polyphase filter implementation of mf
                     * poly_dfil : ndarray
                         polyphase filter implementation of dmf.



                 input      : Complex input signal to the interpolator
                 P          : Upsampling rate of the interpolator
                 q_nom       : (Nominal) downsampling rate of the interpolator

                 -----------------------------------------------------
                 These signals are required if the filter, given --
                 by vector b is not present, they are not used if--
                 b is present                                    --
                 -----------------------------------------------------
            # Outputs:

            out.data
            out.fil
            out.dfil
            out.poly_fil
            out.poly_dfil
            out.pg                : Processing gain of the linear interpolator
            metaSynch.resp
            metaSynch.specMeta
        """

        self.samps_baud_in = samps_baud_in
        self.samps_baud_out = samps_baud_out
        self.P = P
        self.beta = beta  # alpha variable of RRC filter.
        self.taps_per_path = taps_per_path
        self.samp_ratio = samps_baud_out / samps_baud_in
        self.mf = mf
        self.q_nom = np.float(P) * samps_baud_in / samps_baud_out

        self.gen_filters()

    def gen_filters(self):
        """
            Function generates mf, dmf, poly_fil, and poly_dfil

        """
        # Pass in matched filter.
        # if mf is empty then using square pulse shaping.
        # Derive matched filter
        if self.mf is None:
            fil_length = self.P * (self.taps_per_path)
            # need to make the filter odd length
            fil_length += (1 - (fil_length % 2))
            self.mf = make_rrc_filter(self.beta, fil_length, self.P * self.samps_baud_in)
            # pad truncate to make filter length = self.P 
            self.mf = self.mf[:-1]
            self.mf = self.mf / max(self.mf)
            # build protoype signal shaping filter.
        else:
            fil_length = len(self.mf)

        poly_fil = np.reshape(self.mf, (self.P, -1), order='F')
        # compute differential filter.
        # need lines for HW
        fil_gain = max(np.sum(poly_fil, 1))

        self.poly_fil = poly_fil / fil_gain
        self.mf = np.reshape(self.poly_fil, (-1,), order='F')

        diff_filter = [1, 0, -1]
        taps = np.zeros(np.shape(self.mf))

        # mimicking gnuradio.
        diff_filter2 = np.flipud(diff_filter)
        for ii in range(len(self.mf)):
            for (jj, diff_val) in enumerate(diff_filter2):
                index = (ii - 1 + jj) % len(self.mf)
                taps[ii] = taps[ii] + diff_val * self.mf[index]  # ii-1+jj]

        pwr = np.sum(np.abs(taps))
        mf_pwr = np.sum(np.abs(self.mf))
        # compare max pwr and scale differential to have equivalent maximum
        # gain.
        gain_lin = mf_pwr / pwr

        self.dmf = taps * gain_lin
        self.dmf_orig = self.dmf
        self.poly_dfil = np.reshape(self.dmf, (self.P, -1), order='F')

        # need lines for HW
        self.fil_gain = fil_gain
        # regenerate dmf

    def gen_fix_filters(self, qvec_coef, qvec, qvec_out=None):
        """
            Helper function that converts matched filter, MF, and differential matched filter, dMF, into fixed point 
            implementation
        """
        qvec_out = qvec if qvec_out is None else qvec_out
        mf_fi, mf_msb, max_tuple, taps_gain = gen_fixed_poly_filter(self.poly_fil, qvec_coef, qvec, qvec_out, self.P)

        delta_gain = max_tuple[1] * taps_gain

        # scale the dmf_fi by the delta_gain applied to mf -- keeps scaling of orignal design.
        # print("delta_gain = {}".format(delta_gain))
        dmf_fi = sfi(self.poly_dfil*delta_gain, qvec_coef)

        return mf_fi, dmf_fi, mf_msb

    def comp_pg(self):
        """
            Function computes the processing gain of the polyphase filter
            banks.
        """

        pg = np.zeros((np.shape(self.poly_fil)[0],))
        # Estimate processing gain of combined filter.
        for (ii, val) in enumerate(self.poly_fil):
            pg[ii] = sum(val)**2. / sum(val**2.)

        return pg

    @staticmethod
    @njit(int64[:](complex128[::1], complex128[:,:,:], int64, int64, float64[::1], float64[::1]))
    def ted_inner_loop(chunk, ted_array, samps_baud_in, P, poly_fil, poly_dfil):
        taps_per_phase = len(poly_fil) // P
        taps_i = np.zeros((taps_per_phase,))
        taps_q = np.zeros((taps_per_phase,))
        rindex = np.zeros((samps_baud_in,), dtype=np.int64)

        for i, value in enumerate(chunk):
            
            offset = i % samps_baud_in    
            taps_i[1:] = taps_i[:-1]
            taps_q[1:] = taps_q[:-1]
            taps_i[0] = np.real(value)
            taps_q[0] = np.imag(value)
            for q_val in range(P):
                lidx = q_val * taps_per_phase
                ridx = (q_val + 1) * taps_per_phase
                hmf = poly_fil[lidx:ridx]  
                hbar = poly_dfil[lidx:ridx]

                i_h = np.dot(hmf, taps_i)
                q_h = np.dot(hmf, taps_q)

                mag = np.abs(i_h + 1j* q_h)

                i_dh = np.dot(hbar, taps_i)
                q_dh = np.dot(hbar, taps_q)

                err = (i_h * i_dh + q_h * q_dh)
                ted_array[q_val, offset, rindex[offset]] = err + 1j*mag
            rindex[offset] += 1

        return rindex

    def est_ted_gain(self, input_stream, plot_on=False, dpi=300):
        """
            Generates Timing Error "S Curve" to compute Kpd gain value for a specific input and MF / dMF pair.
        """
        chunk_size = 512 * self.samps_baud_in
        input_stream = input_stream[: -(len(input_stream) % chunk_size)]
        chunks = np.reshape(input_stream, (-1, chunk_size), order='C')
        
        plot_array = []
        shift_array = []
        poly_fil = np.reshape(self.poly_fil, (1, -1)).flatten()
        poly_dfil = np.reshape(self.poly_dfil, (1, -1)).flatten()
        for i, chunk in enumerate(chunks):
            ted_array = np.zeros((self.P, self.samps_baud_in, len(chunk)), dtype=np.complex128)
            rindex = PolyTimingRec.ted_inner_loop(chunk, ted_array, self.samps_baud_in, self.P, poly_fil, poly_dfil)
            temp_err = [] 
            temp_mag = [] 
            for i in range(self.samps_baud_in):
                mean_err = np.mean((np.real(ted_array[:, i, :rindex[i]].T)), axis=0)
                mean_pwr = np.mean((np.imag(ted_array[:, i, :rindex[i]].T)), axis=0)

                # interlace errors and power
                temp_err = np.concatenate((mean_err, temp_err))
                temp_mag = np.concatenate((mean_pwr, temp_mag))

            idx0 = np.where(np.diff(temp_err) < 0)[0]
            idx1 = np.argmin(np.abs(temp_err[idx0]))
            shift = (self.P // 2) * self.samps_baud_in - idx0[idx1]
            # shift so that minimum error with negative slope are aligned.
            #normalize by maximum phase.  This aligns all the phases in terms of error.
            shift_array.append(shift)
            plot_array.append(temp_err)

        # T is symbol period = len(final_array[0])
        # final_array spans 1 symbol period.
        final_array = []
        print("Mean Shift = {}".format(np.mean(shift_array)))
        for row, shift in zip(plot_array, shift_array):
            final_array.append(np.roll(row, shift))

        max_idx = np.argmax(np.max(np.abs(final_array), axis=1))
        max_curve = final_array[max_idx]
        if plot_on:
            num_steps = len(final_array[0])
            start = -.5
            end = .5
            step_size = (1. / num_steps)
            x_vec = [np.arange(start, end, step_size)] * len(final_array)
            title = "TED"
            xlabel = r'$\sf{err}/T$'
            ylabel = r'$\sf{TED\ Output}$'
            # label = ['{}'.format(i) for i in range(np.shape(final_array)[0])]
            plot_time_helper(max_curve, title=title, linestyle=None, linewidth=None, dpi=dpi,
                             savefig=True, plot_on=False, x_vec=x_vec, min_n_ticks=8, xlabel=xlabel, ylabel=ylabel, labelsize=14)

        # estimate Kpd by looking average value of central slope
        shift = int((self.P // 2) * self.samps_baud_in)
        lidx = -int(.05 * self.samps_baud_in * self.P * .5) + shift
        ridx = int(.05 * self.samps_baud_in * self.P * .5) + shift
        slope = (max_curve[lidx] - max_curve[ridx]) /  .1 
        
        return np.mean(slope), np.max(max_curve)

    def comp_fixed_pt(self, qvecCoef=(25, 20), qvec=(16, 15), err_offset=0, PISumBits=18, ncoBits=18, qvec_output=None, qvec_ki=None,
                      qvec_kp=None):
        """
            Function generates fixed point constants
        """


    def sync_fast(self, in_vec, loop_eta=(np.sqrt(2) / 2), loop_bw_ratio=.005, kpd=.004,
                  pll_count_init=None, open_loop=False, loop_delay=0, loop_fil_const=None,
                  accum_init=None, taps_i=None, taps_q=None, loop_reg=None, q_start=0):
        """
            Function runs the synchronization routine
            (floating point simulation)

            ==========
            Parameters
            ==========

                * q_start : int
                    User can specify the starting position of the polyphase
                    filter arm.  Used for re-entry of the synch loop between
                    blocks of samples.
                * looEta : float
                    Synch loop damping ratio.
                * loop_bw_ratio : float
                    Synch loop bandwidth ratio.
        """

        num_outputs = int(3 * len(in_vec) * self.samp_ratio)
        mfil_out = np.zeros((num_outputs,), dtype=np.complex128)
        dmfil_out = np.zeros_like(mfil_out, dtype=np.complex128)
        syms = np.zeros_like(mfil_out, dtype=np.complex128)
        timing_sig = np.zeros_like(mfil_out, dtype=np.int64)
        nco_sig = np.zeros_like(mfil_out, dtype=np.float64)
        loop_sig = np.zeros_like(mfil_out, dtype=np.float64)
        loop_int = np.zeros_like(mfil_out, dtype=np.float64)
        err_sig = np.zeros_like(mfil_out, dtype=np.float64)

        if taps_i is None:
            taps_i = np.zeros((np.shape(self.poly_fil)[1],))
            taps_q = np.zeros((np.shape(self.poly_fil)[1],))

        # taps = taps_i + 1j * taps_q
        if accum_init is None:
            accum_init = 0

        if pll_count_init is None:
            pll_count_init = self.samps_baud_out - 1

        # scaling by self.P * self.samps_baud_in since kpd is given in units of a fraction of the symbol period, T.
        # This normalizes the loop appropriately
        kpd_int = kpd / (self.P * self.samps_baud_in)  # * kpd

        # flatten poly_fil and poly_df
        # ipdb.set_trace()
        poly_fil = np.reshape(self.poly_fil, (1, -1)).flatten()
        poly_dfil = np.reshape(self.poly_dfil, (1, -1)).flatten()
        print("max filters = {} {}".format(np.max(np.abs(poly_fil)), np.max(np.abs(poly_dfil))))

        #  PLL parameters
        (kp, ki) = gain_calc_walls(loop_eta, loop_bw_ratio, kpd_int)
        print("Kp/Ki = {}, {}".format(kp, ki))
        idx_vec = PolyTimingRec.numba_sync(np.array(in_vec), poly_fil, poly_dfil, taps_i, taps_q, mfil_out, dmfil_out,
                                           syms, timing_sig, nco_sig, loop_sig, loop_int, err_sig, kp, ki, self.P, self.samps_baud_in, 
                                           self.samps_baud_out, accum_init, pll_count_init, q_start)


        syms = syms[:idx_vec[0]]
        mfil_out = mfil_out[:idx_vec[1]]
        dmfil_out = dmfil_out[:idx_vec[1]]
        timing_sig = timing_sig[:idx_vec[2]]
        nco_sig = nco_sig[:idx_vec[3]]
        loop_sig = loop_sig[:idx_vec[0]]
        loop_int = loop_int[:idx_vec[0]]
        err_sig = err_sig[:idx_vec[0]]

        return syms, mfil_out, dmfil_out, timing_sig, nco_sig, loop_sig, loop_int, err_sig

    # [::1] indicates a contiguous array avoiding the copy inside np.dot
    @staticmethod
    @njit(int64[:](complex128[:], float64[::1], float64[::1], float64[::1], float64[::1], complex128[::1], complex128[::1], 
                   complex128[::1], int64[:], float64[::1], float64[::1], float64[::1], float64[::1], float64, float64, 
                   int64, int64, int64, float64, int64, int64), cache=False)
    def numba_sync(in_vec, poly_fil, poly_dfil, taps_i, taps_q, mfout, dmfout, syms, timing_sig, nco_sig, loop_sig, loop_int,
                   err_sig, kp, ki, P, samps_baud_in, samps_baud_out, accum_init, pll_count_init, q_start):
        """
            Numba accelerated sync_float function.  Returns a array of integers indicating the lengths of :
            mfil_out (matched fi lter output), syms (synchronized symbols), timing_sig
            
            Function runs the synchronization routine
            (floating point simulation)

            ==========
            Parameters
            ==========

                * q_start : int
                    User can specify the starting position of the polyphase
                    filter arm.  Used for re-entry of the synch loop between
                    blocks of samples.
                * looEta : float
                    Synch loop damping ratio.
                * loop_bw_ratio : float
                    Synch loop bandwidth ratio.
        """
        q_nom = np.float64(P) * samps_baud_in / samps_baud_out
        taps_per_phase = len(poly_fil) // P
        q_start = q_start % P
        # tLock
        # tPullin
        num_samps = len(in_vec)
        # expected number of samples
        mfout_cnt = 0
        sym_cnt = 0
        timing_sig_cnt = 0
        nco_cnt= 0

        ii = 0
        pll_count = pll_count_init
        q_new = q_start
        accum = accum_init

        # simulates real-world loop delay caused by logic latency.
        loop_fil_delay = np.zeros((50,), dtype=np.float64)
        loop_fil = 0
        # Filter Real Samples
        while 1:
            # Grab fractional portion and store it  Update with new fractional skipping operation
            while q_new >= float(P):
                if ii > num_samps - 1:
                    break
                q_new -= np.float64(P)
                taps_i[1:] = taps_i[:-1]
                taps_q[1:] = taps_q[:-1]
                taps_i[0] = np.real(in_vec[ii])
                taps_q[0] = np.imag(in_vec[ii])
                ii += 1
            while (q_new < 0):
                # print("stuffing")
                q_new += np.float64(P)
                taps_i[1:] = taps_i[:-1]
                taps_q[1:] = taps_q[:-1]
                taps_i[0] = np.real(in_vec[ii])
                taps_q[0] = np.imag(in_vec[ii])
            # stuffing operation
            if ii > num_samps - 1:
                break

            nco_sig[nco_cnt] = q_new
            nco_cnt += 1
            q_new = q_new % float(P)
            Q = int(q_new)

            # Pull out filter coefficients
            # Perform Polyphase filtering.
            lidx = Q * taps_per_phase
            ridx = (Q + 1) * taps_per_phase
            hmf = poly_fil[lidx:ridx]  
            hbar = poly_dfil[lidx:ridx]

            i_h = np.dot(hmf, taps_i)
            q_h = np.dot(hmf, taps_q)

            i_dh = np.dot(hbar, taps_i)
            q_dh = np.dot(hbar, taps_q)

            # run pll at 1 sample/baud
            # Now take output and run through PLL run through match filter and
            # differential matched filter.
            if pll_count == 0:
                err = (i_h * i_dh + q_h * q_dh)
                syms[sym_cnt] = i_h + 1j * q_h
                sym_cnt += 1
                prop = kp * err
                accum += ki * err
                if ki == 0:
                    accum = 0
                loop_fil = prop + accum
                timing_sig[timing_sig_cnt] = 1
                loop_sig[sym_cnt] = loop_fil
                loop_int[sym_cnt] = accum
                loop_fil_delay[1:] = loop_fil_delay[:-1]
                loop_fil_delay[0] = loop_fil

                err_sig[sym_cnt] = err
                pll_count = samps_baud_out - 1
                loop_fil = loop_fil_delay[25]
                # average time error for I and Q channel.
                # complex(i_h,q_h); # this goes to the demodulator or
                # constellation plot.
            else:
                timing_sig[timing_sig_cnt] = 0
                err = 0
                pll_count -= 1
                loop_fil_delay[1:] = loop_fil_delay[:-1]
                loop_fil_delay[0] = 0

                loop_fil = loop_fil_delay[25]

            timing_sig_cnt += 1
            # implement PLL algorithm.
            mfout[mfout_cnt] = i_h + 1j * q_h
            dmfout[mfout_cnt] = i_dh + 1j * q_dh
            mfout_cnt += 1

            # feedback controller.
            q_step = loop_fil + q_nom
            q_new += q_step

        return np.array([sym_cnt, mfout_cnt, timing_sig_cnt, nco_cnt])

    def sync_float(self, in_vec, loop_eta=(np.sqrt(2) / 2), loop_bw_ratio=.005, kpd=.004,
                   pll_count_init=None, open_loop=False, loop_delay=0, loop_fil_const=None,
                   accum_init=None, taps_i=None, taps_q=None, loop_reg=None, q_start=0):
        """
            Function runs the synchronization routine
            (floating point simulation)

            ==========
            Parameters
            ==========

                * q_start : int
                    User can specify the starting position of the polyphase
                    filter arm.  Used for re-entry of the synch loop between
                    blocks of samples.
                * looEta : float
                    Synch loop damping ratio.
                * loop_bw_ratio : float
                    Synch loop bandwidth ratio.
        """

        q_nom = np.float(self.P) * self.samps_baud_in / self.samps_baud_out
        q_start = q_start % self.P

        # scaling by self.P * self.samps_baud_in since kpd is given in units of a fraction of the symbol period, T.
        # This normalizes the loop appropriately
        kpd_int = kpd / (self.P * self.samps_baud_in)  # * kpd

        #  PLL parameters
        (kp, ki) = gain_calc_walls(loop_eta, loop_bw_ratio, kpd_int)

        if loop_reg is None:
            loop_reg = np.zeros((loop_delay + 1,))

        # tLock
        # tPullin
        if pll_count_init is None:
            pll_count_init = self.samps_baud_out - 1

        num_samps = len(in_vec)

        # expected number of samples
        if taps_i is None:
            taps_i = np.zeros((np.shape(self.poly_fil)[1],))
            taps_q = np.zeros((np.shape(self.poly_fil)[1],))

        if accum_init is None:
            accum_init = 0

        mfout = []
        dmfout = []
        timing_sig = []
        accum_st = []
        err_st = []
        num_skips_st = []
        loop_fil_st = []
        nco_st = []
        q_step_st = []

        sym_out = []

        ii = 0

        pll_count = pll_count_init
        q_new = q_start

        num_skips = 0
        accum = accum_init

        loop_fil = 0

        # Filter Real Samples'
        # loop_delayCnt = loop_delay;
        while 1:
            # Grab fractional portion and store it  Update with new fractional skipping operation
            while q_new >= float(self.P):
                if ii > num_samps - 1:
                    break
                q_new -= self.P
                # ipdb.set_trace()
                num_skips += 1
                taps_i[1:] = taps_i[:-1]
                taps_q[1:] = taps_q[:-1]
                taps_i[0] = np.real(in_vec[ii])
                taps_q[0] = np.imag(in_vec[ii])
                ii += 1
            # stuffing operation
            while q_new < 0:
                q_new += self.P
                taps_i[1:] = taps_i[:-1]
                taps_q[1:] = taps_q[:-1]
                taps_i[0] = np.real(in_vec[ii])
                taps_q[0] = np.imag(in_vec[ii])

            if ii > num_samps - 1:
                break

            nco_st.append(q_new)
            q_new = q_new % float(self.P)
            Q = int(q_new)

            # Pull out filter coefficients
            # Perform Polyphase filtering.
            hmf = self.poly_fil[Q][::-1]
            hbar = self.poly_dfil[Q][::-1]

            i_h = np.dot(hmf, taps_i)
            q_h = np.dot(hmf, taps_q)

            i_dh = np.dot(hbar, taps_i)
            q_dh = np.dot(hbar, taps_q)

            # run pll at 1 sample/baud
            # Now take output and run through PLL run through match filter and
            # differential matched filter.
            if pll_count == 0:
                err = (i_h * i_dh + q_h * q_dh)
                sym_out.append(i_h + 1j * q_h)
                prop = kp * err
                accum += ki * err
                if ki == 0:
                    accum = 0
                # ipdb.set_trace()
                loop_fil = prop + accum
                timing_sig.append(1)
                pll_count = self.samps_baud_out - 1

                # average time error for I and Q channel.
                # complex(i_h,q_h); # this goes to the demodulator or
                # constellation plot.
            else:
                timing_sig.append(0)
                err = 0
                pll_count -= 1
                loop_fil = 0

            err_st.append(err)

            # implement PLL algorithm.
            mfout.append(i_h + 1j * q_h)
            dmfout.append(i_dh + 1j * q_dh)

            # coming out of the loop filter a "+1" means to advance by one
            # sample.
            # if loop_delay:
            #     loop_reg[1:] = loop_reg[:-1]
            #     loop_reg[0] = loop_fil
            #     loop_fil = loop_reg[-1]

            temp = (1 - open_loop) * loop_fil
            loop_fil_st.append(temp)

            # feedback controller.
            q_step = temp + q_nom
            q_new += q_step

            accum_st.append(accum)
            q_step_st.append(q_step)
            num_skips_st.append(num_skips)

            num_skips = 0

        gain_values = {'kpd':kpd, 'kpd_int':kpd_int, 'kp':kp, 'ki':ki}
        return_dict = {'timing_sig':timing_sig, 'q_step':q_step_st, 'syms':sym_out, 'q_nom':q_nom, 'gain_values': gain_values,
                       'accum': accum_st, 'loop_fil':loop_fil_st, 'nco': nco_st, 'err':err_st, 'num_skips':num_skips_st}

        return return_dict


class BandEdgeFLL(object):

    def __init__(self, beta=.3, tap_cnt=63, sps=4, loop_bw=.01, loop_eta=.7):
        """
            Frequency Locked Loop based on
            "Band Edge Filters Perform Non Data-Aided Carrier and Timing
             Synchronizatio of SDR QAM Receivers"  However, it more
             closely follows GnuRadio's implementation, which generates
             a high and low frequency and the PLL is driven by the energy
             difference between the filter outputs.
        """

        self.sps = sps
        self.loop_eta = loop_eta
        self.loop_bw_ratio = loop_bw
        self.beta = beta
        self.tap_cnt = tap_cnt

        # use a rrc MF as default.
        self.mf = make_rrc_filter(beta, tap_cnt, sps)

        self.kp, self.ki = gain_calc_kpki(self.loop_eta, self.loop_bw_ratio)
        self.gen_filters()

    def gen_filters(self):
        """
            Using two sinc filters for this.
        """
        temp1 = make_sinc_filter(self.beta, self.tap_cnt, self.sps, .5)
        temp2 = make_sinc_filter(self.beta, self.tap_cnt, self.sps, -.5)

        fil = temp1 + temp2
        rot_factor = (1 + self.beta) / self.sps

        self.fil_high = complex_rot(fil, rot_factor)
        self.fil_low = complex_rot(fil, -rot_factor)

    def plot_sigs(self, error, prop, integral, loop, samps, dpi=100, fft_size=256, num_avgs=1,
                  window='rect'):

        ticksize = 8
        titlesize = 12
        labelsize = 9

        gs = plt.GridSpec(5, 2, wspace=.4, hspace=.8)
        fig = plt.figure(figsize=(12, 10))
        fig.set_dpi(dpi)

        ax_error = fig.add_subplot(gs[0, :])
        plot_time_sig(ax_error, error, title=r'$\sf{Loop\ Error}$', labelsize=labelsize, titlesize=titlesize)
        ax_prop = fig.add_subplot(gs[1, 0], sharex=ax_error)
        str_val = r'$\sf{k_{p} * Error}$'
        plot_time_sig(ax_prop, prop, title=str_val, labelsize=labelsize, titlesize=titlesize)
        ax_integral = fig.add_subplot(gs[1, 1], sharex=ax_error)
        str_val = r'$\sf{k_{i} * Error}$'
        plot_time_sig(ax_integral, integral, title=str_val, labelsize=labelsize, titlesize=titlesize)
        ax_loop = fig.add_subplot(gs[2, :], sharex=ax_error)
        plot_time_sig(ax_loop, loop, title=r'$\sf{Loop\ Output}$', labelsize=labelsize, titlesize=titlesize)
        ax_real = fig.add_subplot(gs[3, 0], sharex=ax_error)
        plot_time_sig(ax_real, np.real(samps), title=r'$\sf{Real}$', labelsize=labelsize, titlesize=titlesize)
        ax_imag = fig.add_subplot(gs[3, 1], sharex=ax_error)
        plot_time_sig(ax_imag, np.imag(samps), title=r'$\sf{Imag}$', labelsize=labelsize, titlesize=titlesize)
        ax_water = fig.add_subplot(gs[4, :])
        _, resp_water = waterfall_spec(samps, fft_size, num_avgs, window=window, one_side=False)
        resp_water = resp_water.T

        low_freq = -1
        upper_val = np.shape(resp_water)[1] - 1
        extent_val = [0, upper_val, low_freq, 1]
        y_int = list(range(upper_val + 1))
        if len(y_int) > 15:
            dec_rate = len(y_int) // 15
            y_int = y_int[::dec_rate]

        ax_water.imshow(resp_water, origin='lower', interpolation='nearest', aspect='auto', extent=extent_val,
                        cmap=plt.get_cmap('viridis'))

        ax_water.xaxis.set_ticks(y_int)
        ax_water.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        lat_str = r'$\sf{Discrete\ Frequency\ }$' + r'$\pi$' + r'$\frac{rads}{sample}$'
        ax_water.set_ylabel(lat_str, fontsize=labelsize)
        lat_str = r'$\sf{Spectral\ Slice\ Number}$'
        ax_water.set_xlabel(lat_str, fontsize=labelsize)
        ax_water.tick_params(axis='x', labelsize=ticksize, labelrotation=45)

        ax_water.locator_params(nbins=10)
        ax_water.tick_params(axis='x', labelsize=ticksize)
        ax_water.tick_params(axis='y', labelsize=ticksize)
        title = r'$\sf{Waterfall\ Spectrum}$'
        ax_water.set_title(title, fontsize=titlesize)
        ax_water.grid(False)
        # window='', nperseg=nperseg, noverlap=noverlap, fft_size=fft_size,
        #  return_onesided=return_onesided, normalize=normalize)
        # plot_psd(ax_psd, wvec, resp)
        fig.canvas.draw()
        fig.savefig('./FLL.png')

    def run_faccum_loop(self, in_signal, test_harness=True):
        rot = 0
        buff = RingBuffer(len(self.fil_high), dtype=np.complex)

        fil_high = self.fil_high[::-1]
        fil_low = self.fil_low[::-1]
        int_val = 0
        if test_harness:
            error_ret = []
            prop_ret = []
            int_ret = []
            loop_ret = []
            phasor = []

        ret_samps = []
        for samp in in_signal:
            samp_new = samp * np.exp(1j * np.pi * -rot)
            # push new sample into buffer
            buff.append(samp_new)

            # filters are symmetric
            fil_out_high = np.dot(fil_high, buff.view)
            fil_out_low = np.dot(fil_low, buff.view)

            err_sig = (fil_out_high + fil_out_low) * np.conj(fil_out_high - fil_out_low)

            freq_error = np.real(err_sig)

            prop = self.kp * freq_error
            int_val += self.ki * freq_error

            loop_val = prop + int_val
            ret_samps.append(samp_new)

            # final integration for dds.
            rot += loop_val
            if test_harness:
                error_ret.append(freq_error)
                prop_ret.append(prop)
                int_ret.append(int_val)
                loop_ret.append(loop_val)
                phasor.append(rot)

        if test_harness:
            self.plot_sigs(error_ret, prop_ret, int_ret, loop_ret, ret_samps)

    def run_fll_loop(self, in_signal, test_harness=True):
        rot = 0
        buff = RingBuffer(len(self.fil_high), dtype=np.complex)
        int_val = 0
        if test_harness:
            error_ret = []
            prop_ret = []
            int_ret = []
            loop_ret = []
            phasor = []

        fil_high = self.fil_high[::-1]
        fil_low = self.fil_low[::-1]
        ret_samps = []
        for samp in in_signal:
            samp_new = samp * np.exp(1j * np.pi * rot)
            # push new sample into buffer
            buff.append(samp_new)

            # filters are symmetric
            fil_out_high = np.dot(fil_high, buff.view)
            fil_out_low = np.dot(fil_low, buff.view)

            mag_high = np.abs(fil_out_high)
            mag_low = np.abs(fil_out_low)

            error = mag_low - mag_high

            prop = self.kp * error
            int_val += self.ki * error

            loop_val = prop + int_val
            ret_samps.append(samp_new)

            # final integration for dds.
            rot += loop_val

            if test_harness:
                error_ret.append(error)
                prop_ret.append(prop)
                int_ret.append(int_val)
                loop_ret.append(loop_val)
                phasor.append(rot)

        if test_harness:
            self.plot_sigs(error_ret, prop_ret, int_ret, loop_ret, ret_samps)

        return ret_samps
