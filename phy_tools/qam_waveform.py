# -*- coding: utf-8 -*-
"""
Created on Wed May 25 23:08:25 2016

@author: phil
"""

import numpy as np
import scipy.misc as misc
# import scipy as sp
import matplotlib.pyplot as plt
import pdb
import copy
import dill as pickle
import scipy.signal as signal
import phy_tools.gen_utils as utils
import ipdb

import os
import csv
import time
import sys
from itertools import count

import phy_tools.vhdl_gen as vhdl_gen
import phy_tools.gen_utils as sig_tools
from phy_tools.gen_utils import complex_rot
import phy_tools.fil_utils as fil_utils
import phy_tools.fp_utils as fp_utils
from phy_tools.plt_utils import plot_psd_helper, plot_time_helper
from phy_tools.demod_utils import BandEdgeFLL
from typing import Optional, Tuple, List, Union

from phy_tools import mls

from subprocess import check_output, CalledProcessError, DEVNULL
try:
    __version__ = check_output('git log -1 --pretty=format:%cd --date=format:%Y.%m.%d'.split(), stderr=DEVNULL).decode()
except CalledProcessError:
    from datetime import date
    today = date.today()
    __version__ = today.strftime("%Y.%m.%d")


spb = 4
beta = .05
preamble_len = 64
packet_size = 960
table_bits = 18
dec_primary = 25
cic_width = 18
qvec_in = (cic_width, cic_width - 1)
qvec_adc = (16, 15)
qvec_hilbert = (18, 17)
qvec_coef_final = (18, 17)
qvec_coef_primary = (25, 24)
qvec_coef_hilbert = (18, 17)
m_final_fac = 25
final_fir_msb = 40
plt.style.use('fivethirtyeight')

clk_rate = 96E6  # logic clock rate is 96 MHz.


def ret_linear_map(bit_idx, bit_map):

    pdb.set_trace()


class QAM_Mod(object):

    def __init__(self, plot_on=False, packet_size=960, spb=4, snr=20, preamble_len=0, beta=.3, frame_mod='qpsk',
                 xcode_shift=0, ycode_shift=0, rrc=False):

        self.beta = beta  # using root-raised cosine pulse shaping.

        self.plot_on = plot_on
        self.packet_size = packet_size
        self.spb = spb  # samples per baud.
        self.snr = snr

        self.preamble_len = preamble_len
        if preamble_len:
            n_val = int(np.log2(preamble_len))
            # print("n_val = {}".format(n_val))
            temp = mls.mls(n_val)

            self.preamble = [0 if bool(value) is False else 1 for value in temp]
            self.preamble = self.preamble + [0]
            self.pre_mod_type = 'bpsk'
            self.pre_bit_sym = 1

            map_dict = sig_tools.gen_map(mod_type='bpsk', xcode_shift=1)
            #(bits_per_sym, sym_map, sym_index, bit_map, Es_avg_comp)
            self.pre_bit_sym = map_dict['bits_per_sym']
            self.pre_bit_map = map_dict['bit_map']
            self.pre_sym_map = map_dict['sym_map']

            self.pre_sym_index = sig_tools.ret_sym_index(self.pre_bit_map, self.pre_sym_map)
            self.preamble_len = len(self.preamble)

        n = 32 * self.spb + 1
        # generate pulse shaping filter.
        if rrc:
            self.b_shape = utils.make_rrc_filter(self.beta, n, self.spb)
        else:
            self.b_shape = utils.make_rc_filter(self.beta, n, self.spb)
        self.b_shape = self.b_shape / np.sum(self.b_shape)

        # map into sym_index
        full_path = os.path.dirname(os.path.abspath(__file__))
        file_name = full_path + '/resamp_fil.p'
        try:
            self.resamp_fil = pickle.load(open(file_name, 'rb'))
        except:
            # full_path = os.getcwd()
            fil_utils.gen_resamp_fil(full_path)
            self.resamp_fil = pickle.load(open(file_name, 'rb'))

        self.up_rate = 5000

        self.data_mod_type = frame_mod
        map_dict = sig_tools.gen_map(mod_type=self.data_mod_type, xcode_shift=xcode_shift, ycode_shift=ycode_shift)
        # (bits_per_sym, sym_map, sym_index, bit_map, Es_avg_comp) = result

        self.data_bit_sym = map_dict['bits_per_sym']
        self.data_bit_map = map_dict['bit_map']
        self.data_sym_map = map_dict['sym_map']

        self.data_sym_index = sig_tools.ret_sym_index(self.data_bit_map, self.data_sym_map)

        frame_len = self.packet_size * 8 / self.data_bit_sym
        self.frame_len = frame_len + self.preamble_len
        self.frame_samples = self.frame_len * self.spb

        # create QPSK modulator object.
        self.init_vars()

    def init_vars(self):
        # should replace this with super inside QAM_Demod
        return 0

    def comp_ber_curve(self, const_map, ebno_db=None):
        """
            Method computes BER curve for given constellation.
        """
        num = np.sum(np.abs(const_map)**2)
        den = np.size(const_map)

        num_const_pts = np.size(const_map)
        es_avg = np.sqrt(num / den)
        bit_sym = np.log2(np.size(const_map))

        # Compute BER Curve for given constellation
        # uses Union Bound as an approximation.
        num_rows = np.size(const_map, 0)
        num_cols = np.size(const_map, 1)
        # determine distance to assign to Es
        if ebno_db is None:
            ebno_db = np.arange(-1.6, 25, .01)  # ebno in dB.

        ebno = 10**(ebno_db / 10)  # ebno in Log.
        eb_avg = es_avg / bit_sym
        No = eb_avg / ebno
        noise_var = No
        snr = 10 * np.log10(es_avg / noise_var)

        q_matrix = np.zeros((np.size(const_map), np.size(snr)))

        # Compute Union bound for BER determination.
        # for kk in range(np.size(self.sym_map)):
        kk = 0
        for sub_idx, map_val in np.ndenumerate(const_map):
            i = sub_idx[0]
            j = sub_idx[1]
            # find neighbors
            row_start = i - 1 if i > 0 else 0
            row_end = i + 1 if i < (num_rows - 1) else num_rows - 1
            col_start = j - 1 if j > 0 else 0
            col_end = j + 1 if j < (num_cols - 1) else num_cols - 1
            # create indices --
            temp = np.zeros((1, np.size(self.snr)))
            for mm in np.arange(row_start, row_end + 1):
                for nn in np.arange(col_start, col_end + 1):
                    index = (mm, nn)
                    if index != (i, j):
                        # compute distance
                        dist = abs(const_map[i, j] - const_map[mm, nn])
                        # compute distance as a ratio of Eb.
                        q_value = dist / np.sqrt(2 * No)
                        q_func_out = sig_tools.q_func(q_value)
                        temp = temp + q_func_out

            q_matrix[kk, :] = temp
            kk += 1

        q_matrix = q_matrix / num_const_pts

        # this is probability for symbol error.
        psym_error = np.sum(q_matrix, 0)
        ber_curve = psym_error / bit_sym

        return (ebno_db, snr, ber_curve)

    def comp_per_curve(self, const_map, pkt_length, ebno_db=None, fec_bits=0):
        """
            Helper function generates a PER curve for a given constellation.
            PER=1-(1-BER)^N where N is the number of bits. This is where there
            is no error correction.

            For error correction - able to correct up to m number of bits then.
            You want to add the binomial coefficient since you are finding
            the probability that > m bits were found in error.
            Effectively using Bernoulli Trials theory to calculate PER.
        """
        # pkt_length is in bits.
        (ebno_db, snr, ber_curve) = self.comp_ber_curve(const_map, ebno_db)

        per_curve = np.zeros(np.shape(ber_curve))

        k = pkt_length - fec_bits
        n = pkt_length

        first_term = misc.comb(n, k, exact=1)
        # first term represents the number of combinations that k correct bits
        # can be drawn.
        for ii in range(len(per_curve)):
            # second term represents the probability that  n - m  (k) bits are correct for a given packet.
            # based on uniform distribution of errors and errors are independent.
            sec_term = (1 - ber_curve[ii])**k
            # third term represents the probability that 1 of m bits is in error.
            third_term = ber_curve[ii]**(n - k)
            per_curve[ii] = 1 - first_term * sec_term * third_term

        return (ebno_db, snr, per_curve)

    def conv_to_sym(self, data, sym_index):
        """
            Performs symbol mapping.

            ==========
            Parameters
            ==========

            data          : int
                Binary string of 1's and 0's.

            * sym_index   : symbol mapping order as though input is an index.

            =======
            Returns
            =======

            * syms     : IQ symbols of mapped data
        """
        data_int = data.copy()
        data_int = fp_utils.bin_array_to_uint(np.atleast_2d(data_int.T))

        return sym_index[data_int]

    def gen_syms(self, sym_index, bit_sym, data=None):
        """
            Main generator method for creating symbol map and mapping data onto
            said map.
        """
        data = data
        if data is None:
            np.random.seed(42)
            data = np.round(np.random.rand(1000,)).astype(int)

        # truncate extra data
        ridx = len(data) - np.remainder(len(data), bit_sym)
        data = data[:ridx]

        data_reshape = np.reshape(data, (bit_sym, -1), order='F')

        return self.conv_to_sym(data_reshape, sym_index)

    def gen_syms_w_noise(self, syms, snr, noise_seed=None):

        if noise_seed is None:
            seed_i = int((time.time() % 1) * 1000000000)
            seed_q = int((time.time() % 1) * 1000000000)
        else:
            seed_i = noise_seed
            seed_q = noise_seed + 5

        noise_input = copy.copy(syms)
        if self.snr is not None:
            syms_w_noise, _ = utils.add_noise_pwr(snr, noise_input, seed_i=seed_i, seed_q=seed_q)
        else:
            syms_w_noise = noise_input

        return syms_w_noise

    def gen_shaped_data(self, data, sym_index, bit_sym):
        """
            Helper function applies converts binary data into shaped
            symbols.
        """
        syms = self.gen_syms(sym_index, bit_sym, data)
        return self.gen_shaped_symbols(syms)

    def gen_shaped_symbols(self, syms):
        """
            Helper function applies symbol shaping specified in
            __init__ method.
        """

        shaped_syms = signal.upfirdn(self.b_shape * self.spb, syms, self.spb, 1)

        grp_delay = (len(self.b_shape) - 1) // 2
        lidx = grp_delay
        ridx = -grp_delay
        shaped_syms = shaped_syms[lidx:ridx]

        return shaped_syms

    def resample_symbols(self, symbols, fil_step):
        """
            Resampling, using a nearest neighbor implementation with
            upfirdn
        """
        int_step = np.int(np.round(fil_step))
        symbols = signal.upfirdn(self.up_rate * self.resamp_fil.b, symbols, self.up_rate, int_step)
        # compute offset to update frame starts.
        return symbols, int_step

    @staticmethod
    def freq_shift(symbols, cen_freq):
        """
            Helper function that provides static and sweeping
            frequency shifts of the symbols.
        """
        if cen_freq != 0:
            symbols = utils.complex_rot(symbols, cen_freq)

        return symbols

    def gen_frames(self, num_frames, frame_space_mean=1000, frame_space_std=200,
                   seed=10, cen_freq=0, sig_bw=.5, snr=None, noise_seed=None, retime_percent=None,
                   min_samps=None, sym_snr=None, crc_on=True):

        gaps = (frame_space_std * np.random.standard_normal((num_frames+1,)) + frame_space_mean)

        idx0 = (gaps < 0)
        sig_bw = np.float(sig_bw)
        fil_step = np.int(.5 * self.spb * self.up_rate * sig_bw)

        if snr is None:
            snr = self.snr

        if retime_percent is not None:
            fil_step = int(fil_step * (1 - retime_percent / 100.))

        gaps[idx0] = 0
        # convert this to nominal number of samples
        gap_samps = gaps
        gap_samps = gap_samps.astype(int)

        signal = []
        frame_starts = []
        noise_start = np.zeros((gap_samps[0],))
        for ii in range(num_frames):

            # generate random data -- currently  the data will take all the
            # subcarriers -32 for crc.

            # compute number of bits packet size in number of bytes
            # subtract 32 for CRC and 8 for tail bits.
            num_data_bits = self.packet_size * 8 - 32
            data = utils.gen_rand_data(ii, num_data_bits, dtype=int)
            # prepend a 0
            # calculate crc
            if crc_on:
                crc_val = utils.crc_comp(data, spec='crc32')
                # append CRC bits.
                data_block = np.concatenate((data, crc_val))
            else:
                data_block = data

            # generate symbols
            data_syms = self.gen_syms(self.data_sym_index, self.data_bit_sym, data_block)

            if self.preamble_len:
                pre_syms = self.gen_syms(self.pre_sym_index, self.pre_bit_sym, self.preamble)
                curr_frame = np.concatenate((pre_syms, data_syms))
            else:
                curr_frame = data_syms

            curr_frame = self.gen_shaped_symbols(curr_frame)
            db_val = 10 * np.log10(sig_bw / 2.)
            if sym_snr is not None:
                no_i, no_q = utils.est_no(sym_snr, curr_frame)
            else:
                # this is so after baseband filter you would get the original symbol SNR.
                if snr is not None:
                    no_i, no_q = utils.est_no(snr + db_val, curr_frame)
                else:
                    no_i, no_q = (0., 0.)

            if self.plot_on:
                fig, ax = plt.subplots()
                pre_frame = curr_frame[:256:4]
                data_frame = curr_frame[256::4]
                ax.plot(np.real(pre_frame), np.imag(pre_frame), 'o-')
                ax.plot(np.real(data_frame), np.imag(data_frame), 'go-')
                ax.set_title('TX Frame')
                title = 'Transmitted Signal Constellation'
                fig.canvas.manager.set_window_title(title)
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

            # now add random noise to entire frame.
            # generate a noise vector
            seed_i = 15 + ii
            seed_q = 30 + ii*2
            noise_vals = np.zeros((gap_samps[ii+1],))  #utils.gen_noise(gap_samps[0], no_i, no_q, seed_i, seed_q)
            if ii == 0:
                frame_starts.append(len(noise_vals))
                curr_frame = np.concatenate((noise_start, curr_frame, noise_vals))
            else:
                frame_starts.append(len(signal))
                curr_frame = np.concatenate((curr_frame, noise_vals))

            signal.extend(curr_frame)

        signal, int_step = self.resample_symbols(signal, fil_step)

        noise_vals = utils.gen_noise(len(signal), no_i, no_q, seed_i, seed_q)
        signal += noise_vals
        signal = signal.tolist()

        if min_samps:
            if len(signal) < min_samps:
                # add noise samples
                pad_len = min_samps - len(signal)
                noise_vals = utils.gen_noise(pad_len, no_i, no_q, seed_i, seed_q)
                signal.extend(noise_vals)
            signal = signal[:min_samps]

        # Signal Rotation -- rotate signal to specified centerFreq
        # perform frequency sweeping here if desired.
        delay_adj = self.up_rate / float(int_step)
        signal = QAM_Mod.freq_shift(signal, cen_freq)

        starts = [int(start * delay_adj) for start in frame_starts]

        return (signal, starts)


class QAM_Demod(QAM_Mod):

    def init_vars(self):

        # preamble is bpsk
        self.mini_corr_len = len(self.preamble) / 1
        self.fll_obj = BandEdgeFLL(self.beta, loop_bw=.01)
        # generate preamble sequence.
        return 0

    def gen_preamble_syms(self):
        """
            generates a modulated preamble that can be correlated against.
        """
        return self.gen_preamble_syms_full()[::self.spb]

    def gen_preamble_syms_full(self):
        """
            generates a modulated preamble that can be correlated against.
        """
        return self.gen_shaped_data(self.preamble, self.pre_sym_index, self.pre_bit_sym)

    def preamble_correlation(self, in_vec, avg_len=64, plot_on=False):
        """
            Method performs a stacked correlation.  Used in determing chip
            sync from Preamble sequence.

            ==========
            Parameters
            ==========

            * **in_vec**  : ndarray
                Input vector.

            * **avg_len** : int
                Averager length used for thresholding.


        """
        # corr_seq = upsample(self.gen_preamble_syms(), self.spb)
        # pdb.set_trace()
        # corr_seq = self.gen_preamble_syms()
        corr_seq = self.gen_preamble_syms_full()
        in_vec = in_vec - np.mean(in_vec[:1000])
        if self.plot_on:
            fig, ax = plt.subplots()
            title = 'Constellation Correlation Sequence'
            ax.plot(np.real(corr_seq), np.imag(corr_seq), 'o-')
            ax.set_title(title)
            fig.canvas.draw()
            fig.savefig('../figures/' + title + '.png')

        den = self.mini_corr_len * self.spb
        num_stacks = len(corr_seq) // den
        corr_sects = np.reshape(np.conj(corr_seq), (num_stacks, -1))
        corr_sects = np.fliplr(corr_sects)
        win_len = np.shape(corr_sects)[1]
        # perform minicorrelations
        stack_sum = 0
        for ii, seq in enumerate(corr_sects):
            # all_samples should be false
            temp = np.roll(signal.upfirdn(seq, in_vec), -ii * win_len).flatten()
            stack_sum += np.abs(temp)

        num_samps = avg_len * self.spb
        roll_samps = num_samps / 2
        # avg_val = mov_avg(stack_sum, num_samps)
        avg_val = np.roll(fil_utils.mov_avg(stack_sum, num_samps), -roll_samps)

        if plot_on:
            fig, (ax, ax1, ax2) = plt.subplots(3, sharex=True)
            ax.plot(stack_sum)
            ax.set_title('Corr Output')
            ax1.plot(avg_val)
            ax1.set_title('Correlator Average')
            ax2.plot(np.abs(in_vec))
            ax2.set_title('Input Signal -- Amplitude')
            title = 'Correlator Output'
            fig.canvas.manager.set_window_title(title)
            fig.canvas.draw()
            fig.savefig('../figures/' + title + '.png')

        return (stack_sum[:-num_samps], np.mean(stack_sum), avg_val[:-num_samps], win_len)

    def lms(self, frames, train_symbols, num_taps=5, mu=.005):
        """
            Minimum mean square error equalizer.

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

                * symOut : ndarray
                    vector of equalized symbols

                * wopt : ndarray
                    vector of optimized adaptive filter taps.
        """
        # populate initial p vector -- cross-correlation vector and
        # R Matrix -- autocorrelation matrix.

        # initialize p vector.
        frames_out = []
        for nn, in_vec in enumerate(frames):
            w_taps = np.zeros((num_taps,), np.complex)
            # make center tap 1 + 0j

            offset = num_taps // 2
            w_taps[offset] = 1 + 0j

            ret_vals = []
            # taps_list = []
            for ii in range(10):
                shift_reg = np.zeros((num_taps,), np.complex)
                train_delay = np.zeros((offset - 1,), np.complex)
                for (jj, sym, train_val) in zip(count(), in_vec, train_symbols):
                    # taps_list.append(w_taps)
                    # run transversal filter.
                    y_val = np.dot(np.conj(w_taps), shift_reg)
                    error = train_delay[-1] - y_val

                    w_incr = 2 * mu * np.conj(error) * shift_reg
                    w_taps = w_taps + w_incr

                    shift_reg[1:] = shift_reg[:-1]
                    shift_reg[0] = sym

                    train_delay[1:] = train_delay[:-1]
                    train_delay[0] = train_val

            shift_reg = np.zeros((num_taps,), np.complex)
            for val in in_vec:
                y_val = np.dot(np.conj(w_taps), shift_reg)
                shift_reg[1:] = shift_reg[:-1]
                shift_reg[0] = val
                ret_vals.append(y_val)

            ret_vals = ret_vals[offset - 1:]
            if self.plot_on:
                title = 'Burst No %d LMS Correction' % nn
                limit = 2 * np.max(np.abs(ret_vals))
                fig = plt.figure()
                fig.set_size_inches(8., 6.)
                # fig.subplots_adjust(bottom=.15)
                ax = plt.subplot(211)
                fig.add_axes(ax)
                ax.plot(np.real(ret_vals[64:]), np.imag(ret_vals[64:]), 'x')
                ax.set_ylim(-limit, limit)
                ax.set_xlim(-limit, limit)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(title)
                ax1 = plt.subplot(223)
                ax1.plot(np.real(ret_vals[64:]), 'o-')
                ax1.set_title('I Component')
                ax2 = plt.subplot(224, sharex=ax1)
                ax2.plot(np.imag(ret_vals[64:]), 'ro-')
                ax2.set_title('Q component')
                fig.canvas.manager.set_window_title('Equalized Burst')
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

            frames_out.append(ret_vals)

        return frames_out

    def plot_correlator(self, in_vec, avg_len=64):
        """
            Helper function for plotting the correlator output
        """

        (stack, mean_val, run_avg, _) = self.preamble_correlation(in_vec, avg_len, plot_on=False)

        (fig, (ax0, ax1)) = plt.subplots(2, sharex=True)
        ax0.plot(run_avg)
        ax0.set_title('Running Average')
        ax1.plot(stack)
        ax1.set_title('Stack Correlator Output')
        fig.canvas.draw()

    def find_initial_sync(self, in_vec):
        """
            Method finds the initial sync for decoding.  Currently finds the
            initial sync of the most powerful correlation

            ==========
            Parameters
            ==========

                * in_vec : ndarray
                    Input sample stream.  It will take 2 pn_seq lengths of
                    the input vector to perform cross-correlation.

            =======
            Returns
            =======

                * frame_indices : list
                    List of sample indices indicating the frame start.

        """
        trunc_vec = in_vec
        # first generate stacked correlation to sync to Pilot channel and
        # use correlation peaks and generate a list of sample bundles.
        (stack, mean_val, run_avg, win_len) = self.preamble_correlation(trunc_vec, plot_on=self.plot_on)

        trig_level = self.on_factor * run_avg

        if self.plot_on:
            fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True)
            title = 'Correlator Outputs SNR = {:.0f} Corr Len = {}'.format(self.snr, self.preamble_len)
            ax0.plot(run_avg)
            ax0.set_title('Running Average')
            ax0.set_ylim(0, 1.2 * np.max(run_avg))
            ax1.plot(stack)
            ax1.set_ylim(0, 1.2 * np.max(stack))
            ax1.set_title('Stacked Correlator')
            ax2.plot(trig_level)
            ax2.set_title('Trigger Level')
            ax2.set_ylim(0, 1.2 * np.max(trig_level))
            fig.canvas.manager.set_window_title(title)
            fig.canvas.draw()
            fig.savefig('../figures/' + title + '.png')

        enable_sig, _ = utils.hyst_trigger(trig_level, run_avg, stack)

        # grab the first pulse and find the peak
        max_sig = utils.win_max(enable_sig, stack)
        frame_indices = np.argwhere(max_sig).flatten()
        frame_indices -= (win_len + 1)  # (self.mini_corr_len * self.spb - 1)

        return frame_indices[:1]

    def fitz_freq_corr(self, bursts, num_terms=10):

        print("Using Fitz Frequency Estimator")
        pre_mod = self.gen_preamble_syms()
        # pre_mod = 1. - 2. * np.array(self.preamble)
        pre_len = len(pre_mod)

        term_vec = np.arange(1, num_terms + 1)
        c_val = np.sum(term_vec)

        for ii, burst in enumerate(bursts):
            # derotate preamble symbols using know preamble.
            # the scrambling sequence has already been removed.
            preamble = burst[:pre_len]
            pilot_set = preamble * np.conj(pre_mod)

            terms = []
            for offset in term_vec:
                temp = pilot_set[offset:] * np.conj(pilot_set[:-offset])
                terms.append(temp)

                if self.plot_on:
                    fig, (ax, ax1, ax2) = plt.subplots(3)
                    title = 'Fitz''s Estimator Internal Signals Term %d' % offset
                    ax.plot(pilot_set.real, pilot_set.imag, '+-')
                    ax.set_title('Constellation Pilots - Burst No %d' % ii)
                    ax.set_xlim((-3.5, 3.5))
                    ax1.plot(preamble.real, preamble.imag, '+')
                    ax1.set_title('Preamble Symbols - Burst No %d' % ii)
                    ax1.set_xlim((-3.5, 3.5))
                    ax2.plot(temp.real, temp.imag, '+')
                    ax2.set_xlim((-3.5, 3.5))
                    ax2.set_title('Comp Vector')
                    fig.canvas.manager.set_window_title(title)
                    fig.canvas.draw()
                    fig.savefig('../figures/' + title + '.png')

                    # this is Kay's estimator
                    fig, ax = plt.subplots()
                    title = 'Transmitted Burst - Before Correction'
                    ax.plot(np.real(burst), np.imag(burst), 'x')
                    fig.canvas.manager.set_window_title(title)
                    fig.canvas.draw()
                    fig.savefig('../figures/' + title + '.png')

            fitz_sum = 0
            for term in terms:
                fitz_sum += np.angle(np.sum(term))
                # fitz_sum += np.mean(np.angle(term))
            omega = fitz_sum / c_val
            omega = omega / np.pi
            if self.plot_on:
                print("omega = %f , omega RX = %f".format(omega, omega / self.spb))
                print(omega)
            # if abs(omega) > .000:

            # perform frequency correction
            # burst = utils.complex_rot(burst, -.001 * 4) # -omega)
            burst = utils.complex_rot(burst, -omega)
            # perform phase correction
            # print(cat_array[:pre_len])
            # this is no good.
            new_pilot = burst[:pre_len] * pre_mod
            phase_est_pilot = np.mean(np.angle(new_pilot))
            if self.plot_on:
                (fig, ax0) = plt.subplots()
                ax0.plot(new_pilot.real, new_pilot.imag, 'x')
                fig.canvas.draw()
                print("phase estimate = {}".format(phase_est_pilot))
            rot_factor = np.exp(1j * -phase_est_pilot)

            burst = burst * rot_factor
            bursts[ii] = burst

            if self.plot_on:
                fig, ax = plt.subplots()
                ax.plot(np.real(burst), np.imag(burst), 'x')
                title = 'Transmitted Burst - Fitz''s Correction'
                fig.canvas.manager.set_window_title(title)
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

                limit = 2 * np.max(np.abs(bursts[ii]))

                title = 'Burst No %d Constellation Fitz''s Correction' % ii

                fig = plt.figure()
                fig.set_size_inches(8., 6.)
                # fig.subplots_adjust(bottom=.15)
                ax = plt.subplot(211)
                fig.add_axes(ax)
                ax.plot(bursts[ii].real, bursts[ii].imag, 'x')
                ax.set_ylim(-limit, limit)
                ax.set_xlim(-limit, limit)
                ax.set_title(title)
                ax1 = plt.subplot(223)
                ax1.plot(bursts[ii].real, 'o-')
                ax1.set_title('I Component')
                ax2 = plt.subplot(224, sharex=ax1)
                ax2.plot(bursts[ii].imag, 'ro-')
                ax2.set_title('Q component')
                fig.canvas.manager.set_window_title('Frequency/Phase Corrected Burst')
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

        return bursts

    def kays_freq_corr(self, bursts):

        # print("Using Kay''s Frequency Estimator")
        pre_mod = self.gen_preamble_syms()
        # pre_mod = 1. - 2. * np.array(self.preamble)
        pre_len = len(pre_mod)

        pre_div = pre_len / 2.

        first_term = 1.5 * pre_len / (pre_len ** 2. - 1)
        t_vec = np.arange(0, pre_len - 1)
        sec_term = 1 - ((t_vec - (pre_div - 1)) / pre_div) ** 2
        win = first_term * sec_term

        for ii, burst in enumerate(bursts):
            # derotate preamble symbols using know preamble.
            # the scrambling sequence has already been removed.
            preamble = burst[:pre_len]

            pilot_set = preamble * np.conj(pre_mod)

            # pdb.set_trace()
            comp_vec = pilot_set[1:] * np.conj(pilot_set[:-1])

            if self.plot_on:
                fig, (ax, ax1, ax2) = plt.subplots(3)
                title = 'Kay''s Estimator Internal Signals'
                ax.plot(pilot_set.real, pilot_set.imag, '+-')
                ax.set_title('Constellation Pilots - Burst No %d' % ii)
                ax.set_xlim((-3.5, 3.5))
                ax1.plot(preamble.real, preamble.imag, '+')
                ax1.set_title('Preamble Symbols - Burst No %d' % ii)
                ax1.set_xlim((-3.5, 3.5))
                ax2.plot(comp_vec.real, comp_vec.imag, '+')
                ax2.set_xlim((-3.5, 3.5))
                ax2.set_title('Comp Vector')
                fig.canvas.manager.set_window_title(title)
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

                # this is Kay's estimator
                fig, ax = plt.subplots()
                title = 'Transmitted Burst - Before Correction'
                ax.plot(np.real(burst), np.imag(burst), 'x')
                fig.canvas.manager.set_window_title(title)
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

            kay_est = np.sum(win * np.angle(comp_vec))
            # kay_est = np.mean(np.angle(comp_vec))
            omega = kay_est / np.pi
            if self.plot_on:
                print("omega = %f , omega RX = %f".format(omega, omega / self.spb))
                print(omega)

            # perform frequency correction
            # burst = utils.complex_rot(burst, -.001 * 4) # -omega)
            burst_orig = copy.copy(burst)
            burst = utils.complex_rot(burst, -omega)
            # res_phase = -(omega * np.pi) * len(preamble)
            # burst = complex_rot(burst, -omega, phase=res_phase)
            # pilot_set = complex_rot(pilot_set, -omega)
            # perform phase correction
            # print(cat_array[:pre_len])
            # this is no good.
            new_pilot = burst[:pre_len] * pre_mod
            phase_est_pilot = np.angle(np.mean(new_pilot))
            if self.plot_on:
                print("phase estimate = {}".format(phase_est_pilot))
            rot_factor = np.exp(1j * -phase_est_pilot)

            burst = burst * rot_factor
            bursts[ii] = burst

            if self.plot_on:
                fig, ax = plt.subplots()
                ax.plot(np.real(burst_orig), np.imag(burst_orig), 'x')
                title = 'Transmitted Burst - Before Kays Correction'
                fig.canvas.manager.set_window_title(title)
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

                limit = 2 * np.max(np.abs(bursts[ii]))

                title = 'Burst No %d Constellation After Kays Correction' % ii

                fig = plt.figure()
                fig.set_size_inches(8., 6.)
                # fig.subplots_adjust(bottom=.15)
                ax = plt.subplot(211)
                fig.add_axes(ax)
                ax.plot(bursts[ii].real, bursts[ii].imag, 'x')
                ax.set_ylim(-limit, limit)
                ax.set_xlim(-limit, limit)
                ax.set_title(title)
                ax1 = plt.subplot(223)
                ax1.plot(bursts[ii].real, 'o-')
                ax1.set_title('I Component')
                ax2 = plt.subplot(224, sharex=ax1)
                ax2.plot(bursts[ii].imag, 'ro-')
                ax2.set_title('Q component')
                fig.canvas.manager.set_window_title('Frequency/Phase Corrected Burst')
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

        return bursts

    def gain_corr(self, bursts):

        ret_bursts = []
        for ii, burst in enumerate(bursts):
            gain_val = 1. / np.mean(np.abs(burst))
            ret_bursts.append(gain_val * np.array(burst))

        return ret_bursts

    def pll_loop(self, bursts, init_phase=0,
                 loop_eta=.707, loop_bw_ratio=.005):
        """
            Method implements the baseband version of PLL loop.
            Leverages that the signal is bpsk

            ==========
            Parameters
            ==========

                * bursts : ndarray
                    Data portion of the bursts.
                * pre_bursts :
                    Preamble portion of the bursts

            =======
            Returns
            =======

                * out_vec : ndarray

        """

        kp_pilot, ki_pilot = utils.gain_calc_kpki(loop_eta, loop_bw_ratio)
        kp_data, ki_data = utils.gain_calc_kpki(loop_eta, loop_bw_ratio / 3.)

        # ki_data = ki_pilot / 5.
        # kp_data = kp_pilot / 5.
        pre_mod = self.gen_preamble_syms()
        const = np.sqrt(2) / 2.

        err_st = []
        slice_st = []
        ret_bursts = []
        for jj, burst in enumerate(bursts):
            pre_len = len(pre_mod)
            # cat_array = np.concatenate((pilot_set, burst))
            ret_vals = []
            # integrator always starts a 0.
            int_val = 0
            rot_phase = init_phase
            for (ii, val) in enumerate(burst):
                new_val = val * np.exp(1j * rot_phase * np.pi)
                ret_vals.append(new_val)
                if ii >= pre_len:
                    sliced_sym = const * (np.sign(new_val.real) + 1j * np.sign(new_val.imag))

                    ki = ki_data
                    kp = kp_data
                else:
                    sliced_sym = np.sign(new_val.real)
                    ki = ki_pilot
                    kp = kp_pilot
                # compute error.
                err = (np.imag(new_val) * np.real(sliced_sym) -
                       np.real(new_val) * np.imag(sliced_sym))
                # normalize the error values..
                err /= np.abs(sliced_sym)

                err_st.append(err)
                slice_st.append(sliced_sym)
                # update loop filter
                int_val += ki * -err
                # int_val    = -.000299
                prop_val = kp * -err
                # prop_val = 0
                loop_out = int_val + prop_val
                rot_phase += loop_out
                # print(rot_phase, loop_out, int_val, prop_val, err)

            ret_bursts.append(np.asarray(ret_vals))

            if self.plot_on:
                title = 'Burst No %d PLL Correction' % jj
                limit = 2 * np.max(np.abs(ret_vals))
                fig = plt.figure()
                fig.set_size_inches(8., 6.)
                # fig.subplots_adjust(bottom=.15)
                ax = plt.subplot(211)
                fig.add_axes(ax)
                ax.plot(np.real(ret_vals[64:]), np.imag(ret_vals[64:]), 'x')
                ax.set_ylim(-limit, limit)
                ax.set_xlim(-limit, limit)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(title)
                ax1 = plt.subplot(223)
                ax1.plot(np.real(ret_vals[64:]), 'o-')
                ax1.set_title('I Component')
                ax2 = plt.subplot(224, sharex=ax1)
                ax2.plot(np.imag(ret_vals[64:]), 'ro-')
                ax2.set_title('Q component')
                fig.canvas.manager.set_window_title('Frequency/Phase Corrected Burst')
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

        return ret_bursts

    def slice_sym(self, in_value):
        """
            returns nearest symbol to the input value.  Used for single value.
        """
        temp = np.abs(in_value - self.data_sym_index)
        min_idx = np.argmin(temp)
        # return list of bits.
        return fp_utils.dec_to_list(min_idx, self.data_bit_sym)

    def decode_syms(self, frames):

        # run normalization through
        new_frames = []
        for frame in frames:
            avg_pwr = np.mean(np.abs(frame[:preamble_len]))
            new_frames.append(frame / avg_pwr)

        frames = self.pll_loop(new_frames, loop_eta=.7, loop_bw_ratio=.010)

        # now discard preambles.
        # now run through LMS algorithm
        train_symbols = (self.gen_preamble_syms()).astype(np.complex)
        frames = self.lms(frames, train_symbols, num_taps=7, mu=.010)

        payloads = []
        pre_len = len(self.preamble)
        for frame in frames:
            payloads.append(frame[pre_len:])

        # now perform hard demodulation of payloads
        # compute Euclidian distances of symbols
        decode_blocks = []
        for payload in payloads:
            bits = []
            for sym in payload:
                testa = self.slice_sym(sym)
                bits.extend(testa)
            print(fp_utils.list_to_hex(bits[:32]))
            decode_blocks.append(bits)

    def decode_syms_simple(self, frames):

        # run normalization through
        new_frames = []
        for frame in frames:
            avg_pwr = np.mean(np.abs(frame[:preamble_len]))
            new_frames.append(frame / avg_pwr)

        new_frames = []
        for frame in frames:
            preamble = frame[:self.preamble_len]
            new_preamble = []
            pdb.set_trace()
            for (sym, pre_sym) in zip(preamble, self.preamble):
                if pre_sym == 1:
                    sym = -1 * sym
                new_preamble.append(sym)
            avg_val = np.mean(new_preamble)
            phase = np.angle(avg_val)
            new_frame = frame * np.exp(-1j * phase)
            new_frames.append(new_frame)

        # now discard preambles.
        # now run through LMS algorithm

        payloads = []
        pre_len = len(self.preamble)
        for frame in new_frames:
            payloads.append(frame[pre_len:])

        # now perform hard demodulation of payloads
        # compute Euclidian distances of symbols
        decode_blocks = []
        for payload in payloads:
            bits = []
            for sym in payload:
                testa = self.slice_sym(sym)
                bits.extend(testa)
            print(fp_utils.list_to_hex(bits[:32]))
            decode_blocks.append(bits)

    def decode(self, in_vec, on_factor=6, use_fll=False):
        # determine frame sync.
        # in_vec = self.coarse_corr.run_fll_loop(in_vec)

        in_vec = sig_tools.complex_rot(in_vec, -.25)
        self.on_factor = on_factor
        frame_idx = self.find_initial_sync(in_vec)

        if self.plot_on:
            print("# of frames = {}, frame_indices = {}".format(len(frame_idx), frame_idx))

        # break up into frames. Have known lengths.
        frames = []
        # fil_offset = ((len(self.b_shape) - 1) // 2) // self.spb
        for start_idx in frame_idx:
            lidx = start_idx
            ridx = lidx + self.frame_samples
            curr_frame = in_vec[lidx:ridx]
            NFFT = 1024
            win = signal.blackmanharris(NFFT)

            if self.plot_on:
                fig_psd, ax_psd = plt.subplots()
                ax_psd.psd(curr_frame, NFFT=NFFT, Fs=2, window=win, noverlap=768,
                           pad_to=512, sides='twosided',
                           scale_by_freq=True)
                ax_psd.set_title('Spectrum Before FLL')

                fig_time, (ax_x, ax_y) = plt.subplots(2, sharex=True)
                ax_x.plot(np.real(curr_frame), 'o-')
                ax_y.plot(np.imag(curr_frame), 'o-')
                fig_time.canvas.set_window_title('Current Frame - Time')
                fig_time.canvas.draw()

            if use_fll:
                limit = 1.5
                title = 'Burst Before FLL Correction'
                fig = plt.figure()
                fig.set_size_inches(8., 6.)
                # fig.subplots_adjust(bottom=.15)
                ax = plt.subplot(211)
                fig.add_axes(ax)
                ax.plot(np.real(curr_frame), np.imag(curr_frame), 'x')
                ax.set_ylim(-limit, limit)
                ax.set_xlim(-limit, limit)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(title)
                ax1 = plt.subplot(223)
                ax1.plot(np.real(curr_frame), 'o-')
                ax1.set_title('I Component')
                ax2 = plt.subplot(224, sharex=ax1)
                ax2.plot(np.imag(curr_frame), 'ro-')
                ax2.set_title('Q component')
                fig.canvas.manager.set_window_title('Before FLL Burst')
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

                curr_frame = self.fll_obj.run_fll_loop(curr_frame,
                                                       test_harness=self.plot_on)

            if self.plot_on and use_fll:
                title = 'Burst After FLL Correction'
                fig = plt.figure()
                fig.set_size_inches(8., 6.)
                # fig.subplots_adjust(bottom=.15)
                ax = plt.subplot(211)
                fig.add_axes(ax)
                ax.plot(np.real(curr_frame), np.imag(curr_frame), 'x')
                ax.set_ylim(-limit, limit)
                ax.set_xlim(-limit, limit)
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(title)
                ax1 = plt.subplot(223)
                ax1.plot(np.real(curr_frame), 'o-')
                ax1.set_title('I Component')
                ax2 = plt.subplot(224, sharex=ax1)
                ax2.plot(np.imag(curr_frame), 'ro-')
                ax2.set_title('Q component')
                fig.canvas.manager.set_window_title('Before FLL Burst')
                fig.canvas.draw()
                fig.savefig('../figures/' + title + '.png')

                # fig, ax = plt.subplots()
                ax_psd.psd(curr_frame, NFFT=NFFT, Fs=2, window=win,
                           noverlap=768,
                           pad_to=512, sides='twosided',
                           scale_by_freq=True)
                ax.set_title('Spectrum After FLL')
                fig_psd.canvas.draw()

            # p = Periodogram(curr_frame, sampling=1024)
            # p.run()
            # pass frame through pulse shaping filter.
            curr_rsh = np.reshape(curr_frame[:600], (-1, 4))
            sum_val = np.sum(curr_rsh, axis=1)

            # fil_out = signal.upfirdn(self.b_shape, curr_frame, 1, self.spb)
            # fil_out = fil_out[fil_offset:-fil_offset]

            fig_time, (ax_x, ax_y) = plt.subplots(2, sharex=True)
            ax_x.plot(np.real(sum_val[64:]), 'o-')
            ax_y.plot(np.imag(sum_val[64:]), 'o-')
            ax_x.set_title('QPSK Symbols')
            fig_time.canvas.set_window_title('Current Frame - MF')
            fig_time.canvas.draw()

            frames.append(sum_val)

        for ii, ret_vals in enumerate(frames):
            title = 'Burst No %d Matched Filter' % ii
            limit = 2 * np.max(np.abs(ret_vals))
            fig = plt.figure()
            fig.set_size_inches(8., 6.)
            # fig.subplots_adjust(bottom=.15)
            ax = plt.subplot(211)
            fig.add_axes(ax)
            ax.plot(np.real(ret_vals[64:]), np.imag(ret_vals[64:]), 'x')
            ax.set_ylim(-limit, limit)
            ax.set_xlim(-limit, limit)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(title)
            ax1 = plt.subplot(223)
            ax1.plot(np.real(ret_vals[64:]), 'o-')
            ax1.set_title('I Component')
            ax2 = plt.subplot(224, sharex=ax1)
            ax2.plot(np.imag(ret_vals[64:]), 'ro-')
            ax2.set_title('Q component')
            fig.canvas.manager.set_window_title(title)
            fig.canvas.draw()
            fig.savefig('../figures/' + title + '.png')

#        fig_time, (ax_x, ax_y) = plt.subplots(2, sharex=True)
#        ax_x.plot(np.real(frames[0]), 'o-')
#        ax_y.plot(np.imag(frames[0]), 'o-')
#        fig_time.canvas.set_window_title('Current Frame - Symbols')
#        fig_time.canvas.draw()

        # frames still have preamble attached.
        # frames = self.kays_freq_corr(frames)
        # frames = self.fitz_freq_corr(frames)

        # use PLL to perform fine frequency correction.
        frames = self.pll_loop(frames, loop_eta=.7, loop_bw_ratio=.010)

        # now discard preambles.
        # now run through LMS algorithm
        train_symbols = (self.gen_preamble_syms()).astype(np.complex)
        frames = self.lms(frames, train_symbols, num_taps=7, mu=.010)

        payloads = []
        pre_len = len(self.preamble)
        for frame in frames:
            payloads.append(frame[pre_len:])

        # now perform hard demodulation of payloads
        # compute Euclidian distances of symbols
        decode_blocks = []
        for payload in payloads:
            bits = []
            for sym in payload:
                testa = self.slice_sym(sym)
                bits.extend(testa)
            print(fp_utils.list_to_hex(bits[:32]))
            decode_blocks.append(bits)

        # now perform CRC check on bits.
        crc_checks = []
        for block in decode_blocks:
            # check crc.
            crc_val = block[-32:]
            crc_check = utils.crc_comp(block[:-32], spec='crc32')
            if crc_val == crc_check:
                crc_checks.append(0)  # 0 indicates crc passed
            else:
                crc_checks.append(1)  # 1 indicates crc failed.

        return crc_checks, frame_idx


def plot_theory(mod_obj, pkt_size):

    ebno_db = np.arange(-1.6, 14, .1)

    ebno, snr_vec, ber = mod_obj.comp_ber_curve(mod_obj.data_sym_map,
                                                ebno_db=ebno_db)

    fig, ax = plt.subplots()
    ax.semilogy(snr_vec, ber)
    title = 'BER Plot'
    ax.set_title(title)
    ax.set_ylabel('BER')
    ax.set_xlabel('SNR - dB')
    ax.set_ylim([.0000001, 10])
    fig.canvas.manager.set_window_title(title)
    fig.canvas.draw()

    ebno, snr_vec, per = mod_obj.comp_per_curve(mod_obj.data_sym_map,
                                                pkt_size,
                                                ebno_db=ebno_db)

    fig, ax = plt.subplots()
    ax.semilogy(snr_vec, per)
    title = 'PER Plot'
    ax.set_title(title)
    ax.set_ylabel('PER')
    ax.set_xlabel('SNR - dB')
    ax.set_ylim([.0000001, 10])
    fig.canvas.manager.set_window_title(title)
    fig.canvas.draw()


def plot_detector_results(mod_obj, pkt_size, snr_val):

    snr = []
    missed_frames = []
    crc_fails = []
    total_packets = []
    per = []
    tot_time = []
    packet_size = []
    preamble_len = []
    threshold = []
    false_pos = []
    false_neg = []

    # extract data from .csv file.
    with open('../test_results/test_detector_results.csv', 'rb') as fp:
        reader = csv.reader(fp, delimiter=',')
        # skip header for
        for ii, row in enumerate(reader):
            if ii == 0:
                continue
            try:
                snr.append(float(row[0]))
                missed_frames.append(int(row[1]))
                crc_fails.append(int(row[2]))
                total_packets.append(int(row[3]))
                per.append(float(row[4]))
                tot_time.append(float(row[5]))
                packet_size.append(int(row[6]) * 8)
                preamble_len.append(int(row[7]))
                threshold.append(float(row[8]))
                false_pos.append(int(row[9]))
                false_neg.append(int(row[10]))

            except:
                pdb.set_trace()

    # combine statistics with identical snr values.

    total_packets = np.array(total_packets)
    crc_fails = np.array(crc_fails)
    missed_frames = np.array(missed_frames)
    false_pos = np.array(false_pos)
    false_neg = np.array(false_neg)

    plen_unique = np.unique(preamble_len)
    det_false_p_final = []
    det_false_n_final = []

    for plen_val in plen_unique:
        # return all indices with this snr value.
        idx = np.nonzero(plen_val == preamble_len)[0].tolist()
        tot_packets = np.sum(total_packets[idx])
        # miss_frames = np.sum(missed_frames[idx])
        # temp = miss_frames / float(tot_packets)

        det_false_p_final.append(np.sum(false_pos[idx]) / float(tot_packets))
        det_false_n_final.append(np.sum(false_neg[idx]) / float(tot_packets))

    # replace all 0's with very small number
    det_false_p_final = [10E-20 if val == 0 else val for val in det_false_p_final]
    det_false_n_final = [10E-20 if val == 0 else val for val in det_false_n_final]

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    ax.semilogy(plen_unique, det_false_p_final, 'ro-', label='False Positive Rate')
    ax.semilogy(plen_unique, det_false_n_final, 'go-', label='False Negative Rate')
    title = 'Detector Statistics'
    ax.set_title(title)
    ax.set_ylabel('Detector Statistics  - SNR : %d' % snr_val)
    ax.set_xlabel('Preamble Length (Symbols)')
    ax.set_ylim([.0000001, 10])
    ax.set_xlim([0, 130])
    sig_tools.attach_legend(ax)

    fig.canvas.manager.set_window_title(title)
    fig.savefig('../figures/Detector_Plot.png')
    fig.canvas.draw()


def plot_test_results(mod_obj, pkt_size):

    plot_theory(mod_obj, pkt_size)

    snr = []
    missed_frames = []
    crc_fails = []
    total_packets = []
    per = []
    tot_time = []
    packet_size = []

    # extract data from .csv file.
    with open('../test_results/test_results.csv', 'rb') as fp:
        reader = csv.reader(fp, delimiter=',')
        # skip header for
        for ii, row in enumerate(reader):
            if ii == 0:
                continue
            try:
                snr.append(float(row[0]))
                missed_frames.append(int(row[1]))
                crc_fails.append(int(row[2]))
                total_packets.append(int(row[3]))
                per.append(float(row[4]))
                tot_time.append(float(row[5]))
                packet_size.append(int(row[6]) * 8)
            except:
                pdb.set_trace()

    # combine statistics with identical snr values.

    total_packets = np.array(total_packets)
    crc_fails = np.array(crc_fails)
    missed_frames = np.array(missed_frames)

    snr_unique = np.unique(snr)
    per_final = []

    for snr_val in snr_unique:
        # return all indices with this snr value.
        idx = np.nonzero(snr_val == snr)[0].tolist()
        tot_packets = np.sum(total_packets[idx])
        crc_f = np.sum(crc_fails[idx])
        miss_frames = np.sum(missed_frames[idx])
        per_temp = (crc_f + miss_frames) / float(tot_packets)

        per_final.append(per_temp)

    ebno_db = np.arange(-1.6, 14, .1)
    ebno, snr_vec, per = mod_obj.comp_per_curve(mod_obj.data_sym_map,
                                                pkt_size,
                                                ebno_db=ebno_db)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    ax.semilogy(snr_vec, per, label='Theoretical')
    ax.semilogy(snr_unique, per_final, 'ro-', label='Simulation Results')
    title = 'PER Plot'
    ax.set_title(title)
    ax.set_ylabel('PER')
    ax.set_xlabel('SNR - dB')
    ax.set_ylim([.0000001, 10])
    ax.set_xlim([-5, 25])
    sig_tools.attach_legend(ax)
    fig.canvas.manager.set_window_title(title)
    fig.savefig('../figures/Per_Plot.png')
    fig.canvas.draw()


def copy_mif():

    import glob
    import shutil
    import os

    files = glob.iglob(os.path.join('./cores', "*.mif"))
    for file_name in files:
        if os.path.isfile(file_name):
            shutil.copy2(file_name, '../vhdl/aldec/qpsk_wksp/txrx/{}'.format(file_name[8:]))


def calc_settings(threshold=4, msetting=64, look_back=1500, frame_len=2048):

    msetting_fi = fp_utils.ufi(msetting, (10, 0), signed=0)
    look_back_fi = fp_utils.ufi(look_back, (11, 0), signed=0)
    frame_len_fi = fp_utils.ufi(frame_len, (16, 0), signed=0)

    print("msetting binary = {}".format(fp_utils.dec_to_ubin(msetting_fi.vec[0], msetting_fi.qvec[0])))

    print("look back binary = {}".format(fp_utils.dec_to_ubin(look_back_fi.vec[0], look_back_fi.qvec[0])))

    print("frame len binary = {}".format(fp_utils.dec_to_ubin(frame_len_fi.vec[0], frame_len_fi.qvec[0])))

    print("log table input qvec = {}".format(qvec_in))
    (combined_table, u_table) = sig_tools.make_log_tables(qvec_in, table_bits=table_bits)
    print("log qvec = {}".format(u_table.qvec))

    log_thresh = np.log(threshold)
    log_thresh_fi = fp_utils.sfi(log_thresh, u_table.qvec, signed=1)

    # create file for logic simulation.
    print("threshold linear = {}".format(threshold))
    print("threshold log (fixed integer) = {}".format(log_thresh_fi.vec[0]))
    print("threshold log (binary) = {0:}".format(fp_utils.dec_to_ubin(log_thresh_fi.vec[0], log_thresh_fi.qvec[0])))

    print("threshold log (hex) = {0:}".format(log_thresh_fi.hex))

    return (msetting_fi, look_back_fi, frame_len_fi, combined_table, u_table, log_thresh_fi)


def gen_correlator():

    import shutil

    beta = .1
    path = '/home/phil/git_clones/qpsk_link/test_input/'
    file_name = 'corr_test.bin'

    test_file = path + file_name

    # sig_tools.write_complex_samples(sig_fi.vec, test_file)

    threshold = 6
    # pre_len = 1024
    fil_out_width = 18
    time_bits = 64
    cic_width = 18
    cic_avg_len = 128
    cic_max_len = 255
    circ_buff_len = 2 ** 10

    msetting = 64
    look_back = 1500
    frame_len = 2048

    calc_tuple = calc_settings(threshold, msetting, look_back, frame_len)

#    msetting_fi = calc_tuple[0]
#    look_back_fi = calc_tuple[1]
#    frame_len_fi = calc_tuple[2]
    combined_table = calc_tuple[3]
#    u_table = calc_tuple[4]
#    log_thresh_fi = calc_tuple[5]

    sig = sig_tools.read_complex_samples(test_file)

    sig /= 2.**qvec_adc[1]
    # read test signal
    # self, plot_on=False, packet_size=960,
    #              spb=4, snr=20, preamble_len=64
    demod_obj = QAM_Demod(preamble_len=preamble_len, spb=spb, packet_size=packet_size, beta=beta)

    # corr_seq = demod_obj.gen_preamble_syms()
    # corr_seq = upsample(corr_seq, demod_obj.spb)
    corr_seq = demod_obj.gen_preamble_syms_full()

    print("correlator sequence length = {}".format(len(corr_seq)))
    # corr_seq = corr_seq[:pre_len]
    # demod_obj.corr_seq = corr_seq

    # signal = sig_tools.read_complex_samples(test_file, False, 'h')
    # run this correlator and plot results.
    demod_obj.plot_correlator(sig, avg_len=cic_avg_len)
    qvec_fil_coef = (18, 16)
    qvec_correction = (18, 16)  # carry extra integer bit -- treating table as signed.
    # reduces number of multipliers -- still good dynamic range.

    # convert this to fixed point.
    fil, msb, temp = fil_utils.max_filter_output(corr_seq.real,
                                                 qvec_fil_coef,
                                                 input_width=qvec_adc[0],
                                                 output_width=fil_out_width,
                                                 correlator=True)

    corr_gain = temp[4]
    corr_msb = temp[5]
    print("Correlator Filter MSB {}, Corr Gain = {}, Corr MSB = {}".format(msb, corr_gain, corr_msb))
    # slice down filter
    fil_fi = fp_utils.Fi(np.flipud(fil.vec), qvec=qvec_fil_coef)

    # write this to a .coe file.
    fp_utils.coe_write(fil_fi, radix=10, file_name='./cores/corr_fil.coe', filter_type=True)

    qvec_out = qvec_in
    cic_obj = fil_utils.CICDecFil(M=cic_max_len, N=1, qvec_in=qvec_in, qvec_out=qvec_out)

    corr_gain_fi, offset_fi = cic_obj.gen_tables(qvec_correction=qvec_correction)

    max_offset = np.max(offset_fi.vec)
    fp_utils.coe_write(combined_table, radix=10, file_name='./cores/combined_table.coe')

    # write these values to .coe files.
    fp_utils.coe_write(corr_gain_fi, radix=10, file_name='./cores/corr_gain.coe', filter_type=False)
    fp_utils.coe_write(offset_fi, radix=10, file_name='./cores/offset_table.coe', filter_type=False)

    log_fn = './cores/log_conv.vhd'
    vhdl_gen.gen_log_conv(log_fn, combined_table, type_bits=1, tuser_width=time_bits)

    cic_fn = './cores/cic_top.vhd'

    vhdl_gen.gen_cic_top(cic_fn, cic_obj, count_val=4, qvec_correction=qvec_correction,
                         integr_name='integrator', offset_rom_name='slice_offset_rom',
                         corr_rom_name='corr_rom', corr_mult_name='corr_mult',
                         comb_name='comb', slice_shift=0, max_input=None)

    vhdl_gen.gen_slicer(48, cic_width, cic_width, max_offset, '../vhdl')

    # vhdl_gen.gen_var_delay('../vhdl', depth=cic_max_len // 2, width=cic_width, c_str=c_str)
    vhdl_gen.gen_var_delay('../vhdl', depth=cic_max_len, width=cic_width + time_bits)
    vhdl_gen.gen_var_delay('../vhdl', depth=circ_buff_len, width=time_bits)
    # peak detect delay block.
    vhdl_gen.gen_var_delay('../vhdl', depth=128, width=time_bits)
    vhdl_gen.gen_var_delay('../vhdl', depth=circ_buff_len, width=qvec_adc[0] * 2)



def gen_test_signal():

    path = '/home/phil/git_clones/qpsk_link/test_input/'
    file_name = 'corr_test.bin'

    test_file = path + file_name

    # generate test signal.
    preamble_len = 64
    spb = 4
    packet_size = 960
    snr = 20
    total_packets = 2
    noise_seed = 100
    cen_freq = 0
    qvec_adc = (16, 15)

    mod_obj = QAM_Mod(snr=snr, packet_size=packet_size, preamble_len=preamble_len, spb=spb)
    sig, frame_starts = mod_obj.gen_frames(total_packets, frame_space_mean=1500, frame_space_std=0,
                                           noise_seed=noise_seed, cen_freq=cen_freq)

    sig = sig[15250:]
    sig /= 1.2 * np.max(np.abs(sig))
    sig_fi = fp_utils.Fi(sig, qvec=qvec_adc)

    # store test signal to speed up simulation.
    sig_tools.write_complex_samples(sig_fi.vec, test_file)


def test_detector():
    plot_on = True
    if plot_on is True:
        num_tests = 1
        total_packets = 1
        noise_seed = 100
    else:
        num_tests = 1
        total_packets = 1000
        noise_seed = None

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    # generate packets.
    plt.style.use('fivethirtyeight')
    plt.close('all')
    packet_size = 240
    cen_freq = 0.00

    snr = 8
    preamble_len = 64
    on_factor = 4
    if preamble_len == 32:
        on_factor = 3.2
    elif preamble_len == 64:
        on_factor = 4.
    elif preamble_len == 16:
        on_factor = 3.

    mod_obj = QAM_Mod(snr=snr, packet_size=packet_size, preamble_len=preamble_len)

    tpreamble_len = mod_obj.preamble_len
    demod_obj = QAM_Demod(plot_on=plot_on, packet_size=packet_size, snr=snr, preamble_len=preamble_len)

    for ii in range(num_tests):
        t1 = time.time()

        sig, frame_starts = mod_obj.gen_frames(total_packets, frame_space_mean=1500, frame_space_std=0,
                                               noise_seed=noise_seed, cen_freq=cen_freq)

        # plot PER
        crc_list, frame_starts_rx = demod_obj.decode(sig, on_factor=on_factor)

        frame_starts = np.array(frame_starts)
        # calculate detector misses.
        frame_starts_rx = np.array(frame_starts_rx)

        # calulate False Positives.
        false_pos_list = []
        for value in frame_starts_rx:
            diff = abs(value - find_nearest(frame_starts, value))
            if diff > 2:
                false_pos_list.append(value)
                # remove false positives from frames_starts_rx

        false_pos = len(false_pos_list)
        false_neg_list = []
        for value in frame_starts:
            diff = abs(value - find_nearest(frame_starts_rx, value))
            if diff > 2:
                false_neg_list.append(value)

        false_neg = len(false_neg_list) - len(false_pos_list)

        # calculate PER
        missed_frames = 0
        if len(crc_list) < total_packets:
            missed_frames = total_packets - len(crc_list)

        crc_fails = np.sum(crc_list)
        tot_errors = missed_frames + crc_fails
        # del mod_obj

        per_stat = tot_errors / float(total_packets)
        print(ii, tot_errors, per_stat)

        tot_time = time.time() - t1
        # now compute PER value
        with open('../test_results/test_detector_results.csv', 'ab') as fp:
            writer = csv.writer(fp, delimiter=',')
            data = [[snr, missed_frames, crc_fails, total_packets, per_stat,
                     tot_time, packet_size, tpreamble_len, on_factor, false_pos, false_neg]]
            writer.writerows(data)

    plot_detector_results(mod_obj, packet_size * 8, snr)


def test_run():

    plot_on = True
    if plot_on is True:
        num_tests = 1
        total_packets = 1
        noise_seed = 100
    else:
        num_tests = 1
        total_packets = 2000
        noise_seed = None
    # generate packets.

    plt.close('all')
    packet_size = 240
    cen_freq = 0.003
    on_factor = 6
    preamble_len = 64

    snr = 14

    mod_obj = QAM_Mod(snr=snr, packet_size=packet_size, preamble_len=preamble_len)
    demod_obj = QAM_Demod(plot_on=plot_on, packet_size=packet_size, snr=snr, preamble_len=preamble_len)

    for ii in range(num_tests):
        t1 = time.time()

        sig, _ = mod_obj.gen_frames(total_packets, frame_space_mean=1500, frame_space_std=0,
                                    noise_seed=noise_seed, cen_freq=cen_freq)

        # plot PER
        crc_list, _ = demod_obj.decode(sig, on_factor=on_factor)

        # calculate PER
        missed_frames = 0
        if len(crc_list) < total_packets:
            missed_frames = total_packets - len(crc_list)

        crc_fails = np.sum(crc_list)
        tot_errors = missed_frames + crc_fails

        per_stat = tot_errors / float(total_packets)
        print(ii, tot_errors, per_stat)

        tot_time = time.time() - t1
        # now compute PER value
        with open('../test_results/test_results.csv', 'ab') as fp:
            writer = csv.writer(fp, delimiter=',')
            data = [[snr, missed_frames, crc_fails, total_packets, per_stat, tot_time, packet_size]]
            writer.writerows(data)

    plot_test_results(mod_obj, packet_size * 8)


def test_decode():

    # generate test signal.
    preamble_len = 64
    spb = 4
    packet_size = 960
    snr = 10
    total_packets = 2
    noise_seed = 100
    cen_freq = 0.001
    # qvec_adc = (14, 13)
    plot_on = True

    plt.style.use('fivethirtyeight')
    plt.close('all')

    mod_obj = QAM_Mod(snr=snr, packet_size=packet_size, preamble_len=preamble_len, spb=spb)
    sig, frame_starts = mod_obj.gen_frames(total_packets, frame_space_mean=1500, frame_space_std=0,
                                           noise_seed=noise_seed, cen_freq=cen_freq)

    demod_obj = QAM_Demod(plot_on, packet_size, spb=spb, snr=10, preamble_len=preamble_len)
    demod_obj.decode(sig)


def gen_input():

    from scipy.io import loadmat

    path = "/home/phil/git_clones/qpsk_link/captures/"
    filename = path + "11_18_16_50V_DC_50Ft_2_9kHz.mat"

    # filename = path + "11_17_16 10V DC 50Ft Other room.mat"
    # filename = path + "11_17_16 10V DC 50Ft.mat"
    mat_data = loadmat(filename)
    sig = mat_data['A'].flatten()
    file_len = 30000000

    sig_fi = fp_utils.sfi(sig)

    offset = 490000
    ridx = offset + file_len

    (fig, (ax, ax1)) = plt.subplots(2)
    ax.plot(sig[offset:ridx])
    ax1.plot(sig_fi.vec[offset:ridx])
    fig.canvas.manager.set_window_title('Input Signal')
    fig.canvas.draw()

    # store as binary file to be used with Aldec simulation.
    file_name = "/home/phil/git_clones/qpsk_link/test_input/live_11_18_16.bin"
    sig_tools.write_samples(sig_fi.vec[offset:ridx:2], file_name=file_name)


def load_input(filename):
    from scipy.io import loadmat

    mat_data = loadmat(filename)
    sig = mat_data['A'].flatten()

    idx = np.isinf(sig)
    sig[idx] = 0

    # sig = sig[::2]
    sig /= (1.2 * np.max(np.abs(sig)))

    sig_fi = fp_utils.sfi(sig)
    offset = 490000
    file_len = 30000000
    ridx = offset + file_len

    filen = sig_tools.ret_file_name(filename)

    # store as binary file to be used with Aldec simulation.
    file_name = "/home/phil/git_clones/qpsk_link/test_input/{}.bin".format(filen[:-4])
    sig_tools.write_samples(sig_fi.vec[offset:ridx], file_name=file_name)

    return sig


def gen_final_filter(fc_list, num_taps=625):

    if fc_list is not list:
        fc_list = [fc_list]

    fil_array = np.array([])
    for nn, fc in enumerate(fc_list):
        bb_obj = fil_utils.LPFilter(fc=fc, num_taps=num_taps, trans_bw=.1, M=m_final_fac,
                                    qvec=qvec_hilbert, num_iters=64, num_iters_min=1,
                                    freqz_pts=3000, qvec_coef=qvec_coef_final, quick_gen=True)

        coefs = bb_obj.gen_fixed_filter(desired_msb=final_fir_msb)
        bb_obj.plot_psd(title="Final Filter {}".format(nn), savefig=True)

        fil_array = np.append(fil_array, coefs)

    fil_array_fi = fp_utils.sfi(fil_array, qvec=(qvec_coef_final[0], 0))
    fp_utils.coe_write(fil_array_fi, file_name='./cores/final_fir.coe', filter_type=True)


def filter_specs(sig, fs=500000, baud_rate=20, cen_freq=2900, plot_on=True):

    import phy_tools.plt_utils
    import phy_tools.gen_utils
#    from cycler import cycler
    import glob
    import shutil

#    color_list = ['b', 'b', 'b', 'b', 'g', 'g', 'g', 'g']
#    line_list = ['-', '--', ':', '-.', '-', '--', ':', '-.']
#    marker_list = ['o', 'o', 'o', 'o', 'o', 'o', '.', '.']
    # cycler_dict0 = {0: cycler('color', color_list) + cycler('linestyle', line_list) + cycler('marker', marker_list)}

    threshold = 6

    if plot_on:
        plot_psd_helper(sig, title='Input Spectrum')

    calc_settings(threshold=threshold, msetting=64, look_back=1500, frame_len=2048)

    lp_obj_bb = fil_utils.LPFilter(fc=.05, num_taps=16 * dec_primary, trans_bw=.15, M=dec_primary,
                                   quick_gen=True, qvec_coef=qvec_coef_primary, qvec=qvec_adc, qvec_out=qvec_hilbert)

    lp_obj_bb.gen_fixed_filter(coe_file='./cores/front_fil.coe')
    if plot_on:
        lp_obj_bb.plot_psd(title='Primary Filter', savefig=True)
    print("MSB of Primary Filter = {}".format(lp_obj_bb.msb))

    sig_fil = signal.upfirdn(lp_obj_bb.b, sig, 1, dec_primary)

    lp_obj = fil_utils.LPFilter(fc=.25, num_taps=40, trans_bw=.15, hilbert=True, flip_hilbert=True, qvec=qvec_hilbert,
                                qvec_coef=qvec_coef_hilbert)

    lp_obj.gen_fixed_filter(coe_file='./cores/hilbert_fil.coe')
    if plot_on:
        lp_obj.plot_psd(title='Hilbert Filter', savefig=True)
    print("Hilbert MSB = {}".format(lp_obj.msb))

    # coe file for RX buffer
    buff = range(1, 257)
    buff = buff * 16

    buff_fi = fp_utils.sfi(buff, qvec=(8, 0))
    fp_utils.coe_write(buff_fi, file_name='./cores/rx_buffer.coe')

    buff = range(256)
    buff_fi = fp_utils.sfi(buff, qvec=(9, 0))
    fp_utils.coe_write(buff_fi, file_name='./cores/tx_buffer.coe')

    num_samps = 200000
    sig_hil = lp_obj.hilbert_filter(sig_fil)
    if plot_on:
        plot_psd_helper(sig_fil[:num_samps], title='Decimating Filter Output', savefig=True)
        plot_psd_helper(sig_hil[:num_samps], w_time=False, title='Hilbert Filter Output', savefig=True)

    tune_freq = (cen_freq / (fs / 2.)) * dec_primary * 2
    tune_freq_fi = fp_utils.sfi(-tune_freq, qvec=(48, 47))

    print("tune value = {}".format(tune_freq_fi.hex))
    sig_rot = complex_rot(sig_hil, -tune_freq)
    if plot_on:
        plot_psd_helper(sig_rot[:num_samps], w_time=False, title='Retuned', savefig=True)

    final_dec = int((fs / (baud_rate * spb)) / (dec_primary * 2))
    print("final decimation = {}".format(final_dec))

#    bb_obj = fil_utils.LPFilter(fc=.01, num_taps=final_dec * 5, trans_bw=.1, M=final_dec,
#                                qvec=qvec_hilbert,
#                                qvec_coef=qvec_coef_final, quick_gen=True)
#    sig_bb = signal.upfirdn(bb_obj.b, sig_rot, 1, final_dec)
#    bb_obj.gen_fixed_filter(coe_file='./cores/final_fir.coe')
#
#    print("final msb = {}".format(bb_obj.msb))
#    if plot_on:
#        bb_obj.plot_psd(title="Final Filter", savefig=True)
#        plt_utils.plot_spec_sig(sig_bb[:num_samps], w_time=True, title='Final Decimating Filter Output',
#                              cycler_dict0=cycler_dict0, savefig=True)

    # copy .png files into ../figures
    for file_v in glob.glob(r'*.png'):
        print(file_v)
        shutil.copy(file_v, '../figures')

#    return sig_bb


def gen_spi_commands():

    byte_vals = ['0x14', '0x80', '0x00', '0x01', '0x00', '0x00', '0x00', '0x00', '0x00', '0x00']

    vec = [int(value, 16) for value in byte_vals]

    path = '/home/phil/git_clones/qpsk_link/vhdl/aldec/qpsk_wksp/txrx/'
    file_name = path + 'file0.bin'

    sig_tools.write_pkts(np.array(vec), file_name, format_str='B', append=False)

    # put system into tx_mode
    byte_vals = ['0x01', '0x80', '0x04']
    vec = [int(value, 16) for value in byte_vals]

    # file_name = path + 'file1.bin'
    sig_tools.write_pkts(np.array(vec), file_name, format_str='B', append=True)

    # start Transmit
    byte_vals = ['0x01', '0x80', '0x05']
    vec = [int(value, 16) for value in byte_vals]

#    file_name = path + 'file2.bin'
    sig_tools.write_pkts(np.array(vec), file_name, format_str='B', append=True)

    byte_vals = ['0x0', '0xc0']
#    preamble = seq
#    sof = ["{0:#0{1}x}".format(251, 4)]
#    length = ["{0:#0{1}x}".format(23, 4)]
#    fcd = ["{0:#0{1}x}".format(value, 4) for value in [5, 0]]
#    seq_num = ["{0:#0{1}x}".format(0, 4)]
#    msg_len = ["{0:#0{1}x}".format(126, 4)]
    msg = ["{0:#0{1}x}".format(value, 4) for value in reversed(range(32))]

    byte_vals = byte_vals + msg   # preamble + sof + length + fcd + seq_num + msg_len + msg + crc
    print(byte_vals, len(byte_vals))
    vec = [int(value, 16) for value in byte_vals]

    # path = '/home/phil/git_clones/qpsk_link/test_input/'
#    file_name = path + 'file3.bin'
    sig_tools.write_pkts(np.array(vec), file_name, format_str='B', append=True)

    # byte_vals = ['0x0', '0x40', '0x00']
    byte_vals = ['0x0', '0xc0']
    byte_vals = byte_vals + msg  # + ['0x0'] * 32
    vec = [int(value, 16) for value in byte_vals]

    path = '/home/phil/git_clones/qpsk_link/vhdl/aldec/qpsk_wksp/txrx/'
#    file_name = path + 'file4.bin'
    sig_tools.write_pkts(np.array(vec), file_name, format_str='B', append=True)


def plot_capture(sig, plot_on=False):

    plt.close('all')
    plt.style.use('fivethirtyeight')

    plot_on = True
    preamble_len = 64
    packet_size = 960

    sig_bb = filter_specs(sig, plot_on=plot_on)

    demod_obj = QAM_Demod(plot_on, packet_size, spb=spb, snr=10, preamble_len=preamble_len, beta=beta)
    demod_obj.decode(sig_bb, on_factor=6.0)


def process_raw(file_name):

    import csv

    demod_obj = QAM_Demod(preamble_len=preamble_len, spb=spb, packet_size=packet_size, beta=beta)

    reader = csv.reader(open(file_name), delimiter=' ')

    i_samps = []
    q_samps = []
    for row in reader:
        i_samps.append(int(row[0]))
        q_samps.append(int(row[1]))

    samps = np.array(i_samps) + 1j * np.array(q_samps)
    demod_obj.decode_syms([samps])


def calc_params_opt(baud_rate=20, spb=4, dec_primary=25, cen_freq=2900):

    # we want to maximize sample rate <= 500000, while keeping the final decimation
    # to an integer multiple of 25.
    max_fs = 500000
    mult_fac = 40
    final_dec = m_final_fac

    while 1:
        fs = final_dec * mult_fac * dec_primary * 2 * baud_rate * spb
        if fs <= max_fs:
            break
        else:
            mult_fac -= 1

    final_dec = m_final_fac * mult_fac

    tune_freq = 4. * cen_freq * dec_primary / fs
    tune_freq_fi = fp_utils.sfi(-tune_freq, qvec=(48, 47))
    final_fc = 1. / final_dec

    print("Tune Freq Hex = {}".format(tune_freq_fi.hex))

    return fs, tune_freq_fi, final_dec, final_fc


def calc_params(baud_rate=20, spb=4, final_dec=50, dec_primary=25, cen_freq=2900, fs=None):

    if fs is None:
        fs = final_dec * dec_primary * 2 * baud_rate * spb
    # tune_freq = (cen_freq / (fs / 2.)) * dec_primary * 2.
    tune_freq = 4. * cen_freq * dec_primary / fs
    tune_freq_fi = fp_utils.sfi(-tune_freq, qvec=(48, 47))

    print("Tune Freq Hex = {}".format(tune_freq_fi.hex))

    return fs, tune_freq_fi


def print_params(br_list):

    for br in br_list:
        (fs, _) = calc_params(baud_rate=br, spb=spb, cen_freq=2900)
        print("baud rate = {}, fs = {}".format(br, fs))


def print_params_opt(br_list):

    import csv
    fs_list = []
    final_dec_list = []
    fc_list = []
    fs_param_list = []
    for br in br_list:
        (fs, _, final_dec, fc) = calc_params_opt(baud_rate=br, spb=spb, cen_freq=2900)
        fs_param = int(clk_rate / fs)
        print("baud rate = {}, fs = {}, final dec = {}, fc = {}, fs_param = {}".format(br, fs, final_dec, fc, fs_param))
        fs_list.append(fs)
        fs_param_list.append(fs_param)
        final_dec_list.append(final_dec)
        fc_list.append(fc)

    fc_list_unique = np.unique(fc_list)
    fil_select_list = [np.where(fc_list_unique == fc_value)[0][0] for fc_value in fc_list]
    # find index fc_list_index

    with open('./latex/parameters.csv', 'w') as fp:
        csv_writer = csv.writer(fp, delimiter=',')
        csv_writer.writerow(['Baud Rate', 'Final Decimation', 'Fil Select Param', 'ADC Param', 'Real fs Value'])
        zip_obj = zip(br_list, fs_param_list, final_dec_list, fil_select_list, fs_list)
        for br_param, fs_param, final_dec, fil_select, fs in zip_obj:
            csv_writer.writerow([str(br_param), str(final_dec / m_final_fac), str(fil_select), str(fs_param), str(fs)])

    return fc_list_unique


# if __name__ == "__main__":
#     plt.close('all')
#
# #    snr = 14
# #    mod_obj = QAM_Mod(snr=snr, packet_size=packet_size, preamble_len=preamble_len)
# #    plot_test_results(mod_obj, packet_size * 8)
#
#     gen_spi_commands()
#
#     path = "/home/phil/git_clones/qpsk_link/captures/"
#     filename = path + "1_6_17 BGT to AGT 10 ft away.mat"
#     sig = load_input(filename)
#     filter_specs(sig)
#     br_list = [5, 10, 20, 30, 40, 50, 60, 70, 80]
#     fc_list = print_params_opt(br_list)
#     gen_correlator()
#     gen_test_signal()
#     gen_spi_commands()
#     # filter_specs(sig, plot_on=True)
#     gen_final_filter(fc_list, num_taps=1000)
#     calc_settings()
#     copy_mif()
