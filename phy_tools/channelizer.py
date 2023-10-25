#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:28:03 2016

@author: phil
"""

import scipy as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as manimation
# from matplotlib.animation import MovieWriter
from collections.abc import Iterable

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import copy
import time
import ipdb
import dill as pickle
from collections import OrderedDict
from matplotlib import rc
from argparse import ArgumentParser

import phy_tools.fil_utils as fil_utils
import phy_tools.fp_utils as fp_utils
from phy_tools.chan_utils import calc_fft_tuser_width, gen_chan_top, gen_do_file
from phy_tools.plt_utils import plot_psd_helper, plot_psd, gen_psd, df_str, gen_freq_vec, plot_time_sig
from phy_tools.plt_utils import marker_list
import phy_tools.gen_utils as gen_utils
from phy_tools.gen_utils import upsample, add_noise_pwr, write_complex_samples, read_complex_samples, write_binary_file
from phy_tools.gen_utils import gen_comp_tone, read_binary_file, compass
from phy_tools.qam_waveform import QAM_Mod
from phy_tools.fp_utils import nextpow2

import phy_tools.verilog_gen as vgen
import phy_tools.verilog_filter as vfilter
import phy_tools.adv_pfb as adv_filter
import phy_tools.vgen_xilinx as vgenx

from shutil import copyfile

from subprocess import check_output, CalledProcessError, DEVNULL
try:
    __version__ = check_output('git log -1 --pretty=format:%cd --date=format:%Y.%m.%d'.split(), stderr=DEVNULL).decode()
except CalledProcessError:
    from datetime import date
    today = date.today()
    __version__ = today.strftime("%Y.%m.%d")

#print(plt.style.available)
plt.style.use('seaborn-v0_8')
plt.ion()

dpi = 100

blockl = True

import os
dirname = os.path.dirname(__file__)
GEN_2X = True
IP_PATH = os.path.join(dirname, './chan_test/src/')

if not os.path.isdir(IP_PATH):
    os.makedirs(IP_PATH)

SIM_PATH = os.path.join(dirname, './chan_test/sim/')
if not os.path.isdir(SIM_PATH):
    os.makedirs(SIM_PATH)
# test if path exists
TAPS_PER_PHASE = 32
SIX_DB = 10 * np.log10(.25)
NUM_ITERS = 400
FREQZ_PTS = 20000
PFB_MSB = 43
DESIRED_MSB = PFB_MSB
QVEC = (16, 15)
QVEC_COEF = (25, 24)
M_MAX = 512
FC_SCALE = .85  # was .65
TBW_SCALE = .3
TAPS = None
rc('text', usetex=False)

K_default = OrderedDict([(4, 13.905715942382809), (8, 13.905715942382809), (16, 11.75405975341801), (32, 11.752742309570353), (64, 12.125381164550815), (128, 11.931529541015662), (256, 11.931502990722693), (512, 11.97845520019535), (1024, 11.954858093261757), (2048, 11.954857177734413)])
offset_default = OrderedDict([(4, .5), (8, .5), (16, .5), (32, .5), (64, .5), (128, .5), (256, .5), (512, .5), (1024, .5), (2048, .5)])
msb_default = OrderedDict([(8, 39), (16, 39), (32, 39), (64, 39), (128, 39), (256, 39), (512, 39), (1024, 39), (2048, 39)])

K_terms = OrderedDict([(8, 6.45), (16, 6.330000000000003), (32, 6.333000000000001), (64, 6.326999999999999), (128, 6.297), (256, 6.308999999999998), (512, 6.308999999999998), (1024, 6.308999999999998), (2048, 6.308999999999998)])
msb_terms = OrderedDict([(8, 40), (16, 40), (32, 40), (64, 40), (128, 40), (256, 40), (512, 40), (1024, 40), (2048, 40)])
offset_terms = OrderedDict([(8, 0.5149999999999997), (16, 0.49875), (32, 0.49875), (64, 0.49875), (128, 0.500625), (256, 0.499375), (512, 0.5), (1024, 0.5), (2048, 0.5)])

K_terms = OrderedDict([(8, 32.458509317419505), (16, 32.458509317419505), (32, 32.458509317419505), (64, 28.697231648754062), (
    128, 29.755193951058885), (256, 29.755193951058885), (512, 29.755193951058885), (1024, 29.755193951058885), (2048, 29.755193951058885)])
msb_terms = OrderedDict([(8, 39), (16, 39), (32, 39), (64, 39), (128, 39),
                         (256, 39), (512, 39), (1024, 39), (2048, 39)])
offset_terms = OrderedDict([(8, 0.5), (16, 0.505), (32, 0.505), (64, 0.51), (128, 0.51),
                            (256, 0.5), (512, 0.5), (1024, 0.5), (2048, 0.5)])

K_terms = OrderedDict([(8, 32.458509317419505), (16, 32.458509317419505), (32, 32.458509317419505), (64, 28.697231648754062), (
    128, 7.558755193951058885), (256, 29.755193951058885), (512, 29.755193951058885), (1024, 29.755193951058885), (2048, 29.755193951058885)])
msb_terms = OrderedDict([(8, 39), (16, 39), (32, 39), (64, 39), (128, 39),
                         (256, 39), (512, 39), (1024, 39), (2048, 39)])
offset_terms = OrderedDict([(8, 0.5), (16, 0.5), (32, 0.5), (64, 0.51), (128, 0.51),
                            (256, 0.5), (512, 0.5), (1024, 0.5), (2048, 0.5)])


def ret_qcoef(dsp48e2):
    return (27, 26) if dsp48e2 else (25, 24)

def ret_k_terms(taps_per_phase=24):
    # if TAPS_PER_PHASE == 32:
    #     K = 21.0497
    offset_terms = offset_default
    K_terms = K_default
    msb_terms = msb_default
    if taps_per_phase == 24:
        K_terms = OrderedDict([(8, 13.905715942382809), (16, 11.75405975341801), (32, 11.752742309570353), (64, 12.125381164550815), (128, 11.931529541015662), (256, 11.931502990722693), (512, 11.97845520019535), (1024, 11.954858093261757), (2048, 11.954857177734413)])
        msb_terms = OrderedDict([(8, 39), (16, 39), (32, 39), (64, 39), (128, 39), (256, 39), (512, 39), (1024, 39), (2048, 39)])
        offset_terms = offset_default
    elif taps_per_phase == 16:
        # K_terms = OrderedDict([(8, 5.909457397460962), (16, 6.285075988769546), (32, 6.283267211914079), (64, 6.282817687988298), (128, 6.230665893554707), (256, 6.256452941894549), (512, 6.243489379882831), (1024, 6.243487243652363), (2048, 6.2434869384765825)])
        # msb_terms = OrderedDict([(8, 40), (16, 41), (32, 41), (64, 43), (128, 43), (256, 43), (512, 43), (1024, 43), (2048, 43)])
        K_terms = OrderedDict([(8, 5.909457397460962), (16, 6.285075988769546), (32, 6.283267211914079), (64, 6.282817687988298), (128, 6.230665893554707), (256, 6.256452941894549), (512, 6.243489379882831), (1024, 6.243487243652363), (2048, 6.2434869384765825)])
        msb_terms = OrderedDict([(8, 40), (16, 41), (32, 42), (64, 43), (128, 44), (256, 45), (512, 46), (1024, 47), (2048, 48)])

    return K_terms, msb_terms

rc('text', usetex=False)



class Channelizer(object):
    """
        Implements channelizer class.  It is used to fully design the LPF of the channelizer,
        generate diagnostic plots, and processing of signal streams.

        * It includes both critically sampled and 2X sampled channelizers.

        * M : int  : Number of channels.
        * pbr : float : passband ripple in dB.
        * sba : float : stopband attenuation in dB.
        *
    """
    def __init__(self, M=64, Mmax=None, pbr=.1, sba=-80, taps_per_phase=32, gen_2X=True, qvec_coef=(25, 24),
                 qvec=(18, 17), desired_msb=None, K_terms=K_default, offset_terms=offset_default, fc_scale=1., 
                 tbw_scale=.5, taps=None, max_masks=50, freqz_pts=10_000):

        self.taps_per_phase = taps_per_phase
        self.qvec_coef = qvec_coef
        self.qvec = qvec
        self.max_masks = max_masks

        self.gen_2X = gen_2X
        self.M = M
        self.Mmax = Mmax
        self.Mmax = M if Mmax is None else Mmax

        self.sba = sba
        self.pbr = pbr
        self.desired_msb = desired_msb
        fc = 1. / M
        self.fc = fc * fc_scale
        self.fc_scale = fc_scale
        self.rate = 2 if self.gen_2X else 1
        self.tbw_scale = tbw_scale
        self.freqz_pts = freqz_pts
        self.K = None

        if taps is None:
            self.num_taps = M * taps_per_phase
            taps = self.gen_float_taps(gen_2X, K_terms, offset_terms, M)
        else:
            self.num_taps = len(taps)

        self.gen_fixed_filter(taps, self.desired_msb)

        # generating a 2X filter.
        self.paths = M

    def gen_float_taps(self, gen_2X, K_terms, offset_terms, M):
        self.rate = 1
        if gen_2X:
            self.rate = 2

        self.K = K_terms[M]
        self.offset = offset_terms[M]
        taps = self.tap_equation(M)

        return taps

    def plot_psd(self, fft_size=1024, taps=None, freq_vector=None, title=None, miny=None, pwr_pts=None,
                 freq_pts=None, savefig=False, omega_scale=1, xlabel=df_str):

        """
            Helper function that plot the frequency response of baseband filter.
        """
        h_log, omega = self.gen_psd(fft_size, taps, freq_vector)
        # zoom in on passband
        plot_psd_helper((omega*omega_scale, h_log), title=title, miny=miny, plot_on=True, savefig=savefig, pwr_pts=pwr_pts,
                        freq_pts=freq_pts, xprec=4, xlabel=xlabel, dpi=dpi)

        return 0

    def gen_psd(self, fft_size=1024, taps=None, freq_vector=None, fixed_taps=False):
        """
            Helper generates the PSD of the baseband filter
        """
        if taps is None:
            taps = self.taps_fi if fixed_taps else self.taps

        step = 2. / fft_size
        if freq_vector is None:
            # this vector is normalized frequency.
            freq_vector = np.arange(-1., 1., step)

        omega, h = sp.signal.freqz(taps, worN=freq_vector * np.pi)

        # whole = True
        h_log = 20. * np.log10(np.abs(h))
        h_log -= np.max(h_log)

        omega /= np.pi

        return h_log, omega

    def plot_comparison(self, savefig=False, title=None):
        """
            Method plots the comparison of M/2 and M channelizer filter designs.
        """
        fig = plt.figure()
        plt.tight_layout()
        ax = fig.add_subplot(111)
        miny = -200
        pwr_pts = SIX_DB

        fft_size = 8192

        taps_1x = self.gen_taps(gen_2X=False)
        taps_2x = self.gen_taps(gen_2X=True)

        hlog_1x, omega = self.gen_psd(fft_size, taps_1x)
        hlog_2x, _ = self.gen_psd(fft_size, taps_2x)

        plot_psd(ax, omega, hlog_1x, pwr_pts=None, label=r'$\sf{M Channelizer}$', miny=miny, labelsize=18)
        plot_psd(ax, omega, hlog_2x, pwr_pts=pwr_pts, label=r'$\sf{M/2\ Channelizer}$', miny=miny, labelsize=18)

        if savefig:
            fig.savefig('plot_compare2.png', figsize=(12, 10))
        else:
            fig.canvas.draw()

    def plot_psd_single(self, savefig=False, title=None):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        miny = -200
        pwr_pts = SIX_DB

        fft_size = 8192
        taps_2x = self.gen_taps(gen_2X=True)
        hlog_2x, omega = self.gen_psd(fft_size, taps_2x)
        plot_psd(ax, omega, hlog_2x, pwr_pts=pwr_pts, title=r'$M/2 \sf{\ Channelizer Filter PSD}$', miny=miny, labelsize=20)

        plt.tight_layout()
        if savefig:
            fig.savefig('plot_psd_single.png', figsize=(12, 10), dpi=dpi)
        else:
            fig.canvas.draw()


    def gen_poly_partition(self, taps):
        """
            Returns the polyphase partition of the PFB filter
        """
        return np.reshape(taps, (self.M, -1), order='F')

    def gen_fixed_filter(self, taps, desired_msb=None):
        """
            Generates the fixed-point representation of the PFB filter coefficients
        """
        max_coeff_val = (2**(self.qvec_coef[0] - 1) - 1) * (2 ** -self.qvec_coef[1])

        taps_gain = max_coeff_val / np.max(np.abs(taps))
        taps *= taps_gain
        # M = self.M  #len(taps) // self.taps_per_phase

        taps_fi = (taps * (2 ** self.qvec_coef[1])).astype(np.int32)
        poly_fil = np.reshape(taps_fi, (self.M, -1), order='F')
        max_input = 2**(self.qvec[0] - 1) - 1

        # compute noise and signal gain.
        n_gain = np.max(np.sqrt(np.sum(np.abs(np.double(poly_fil))**2, axis=1)))
        s_gain = np.max(np.abs(np.sum(poly_fil, axis=1)))

        snr_gain = 20. * np.log10(s_gain / n_gain)
        path_gain = s_gain #np.max(np.abs(np.sum(poly_fil, axis=1)))
        bit_gain = nextpow2(np.max(s_gain))

        gain_msb = nextpow2(s_gain)
        max_coef_val = 2.**gain_msb - 1
        in_use = s_gain / max_coef_val

        max_value = np.max(s_gain) * np.max(max_input)
        num_bits = fp_utils.ret_num_bitsS(max_value)
        msb = num_bits - 1

        if in_use > .9:
            new_b = poly_fil
            delta_gain = 1
        else:
            # note we are scaling down here hence the - 1
            msb = msb - 1
            delta_gain = .5 * (max_coef_val / s_gain)
            new_b = np.floor(poly_fil * delta_gain).astype(int)
            s_gain = np.abs(np.max(np.sum(new_b, axis=1)))
            path_gain = np.max(np.abs(np.sum(new_b, axis=1)))
            bit_gain = nextpow2(path_gain)

        poly_fil = new_b
        if desired_msb is not None:
            if msb > desired_msb:
                diff = msb - desired_msb
                poly_fil = poly_fil >> diff
                msb = desired_msb

        taps_fi = np.reshape(poly_fil, (1, -1), order='F').flatten()
        self.taps_fi = taps_fi
        self.poly_fil_fi = poly_fil
        self.poly_fil = poly_fil * (2 ** -self.qvec_coef[1])
        self.taps = taps
        self.taps_per_phase = np.shape(self.poly_fil)[1]
        self.fil_msb = msb
        self.nfft = np.shape(self.poly_fil)[0]
        return (s_gain, n_gain, snr_gain, path_gain, bit_gain)

    @property
    def pfb_msb(self):
        return self.fil_msb

    @staticmethod
    def erfc(x):
        # save the sign of x
        sign = [1 if val >= 0 else -1 for val in x]
        x = np.abs(x)

        # constants
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        ret_val = 1 - sign * y
        return ret_val

    def tap_equation(self, fft_size, K=None, offset=None):
        # using root raised erf function to generate filter prototype
        # less control but much faster option for very large filters.
        # Perfectly fine for standard low-pass filters. Link to code
        # effectively use twiddle algorithm to get the correct cut-off
        # frequency
        # http://www.mathworks.com/matlabcentral/fileexchange/15813-near-
        # perfect-reconstruction-polyphase-filterbank

        if K is None:
            K = self.K

        if offset is None:
            offset = self.offset

        F = np.arange(self.num_taps)
        F = np.double(F) / len(F)

        MTerm = np.round((self.fc_scale * (1. / fft_size))**-1)

        x = K * (MTerm * F - offset)
        A = np.sqrt(0.5 * Channelizer.erfc(x))

        N = len(A)
        idx = np.arange(N // 2)

        A[N - idx - 1] = np.conj(A[1 + idx])
        A[N // 2] = 0

        # this sets the appropriate -6.02 dB cut-off point required for the channelizer
        db_diff = SIX_DB - 10 * np.log10(.5)
        exponent = 10 ** (-db_diff / 10.)

        A = A ** exponent

        b = np.fft.ifft(A)
        b = (np.fft.fftshift(b)).real
        b /= np.sum(b)

        return b

    def gen_fil_params(self, start_size=8, end_size=4096, K_init=13., fc_scale=FC_SCALE, tbw_scale=TBW_SCALE):
        """
            Determines optimum K values and required MSB values for varying FFT sizes, given filter
            corner frequency and transition bandwidth.
        """

        end_bits = int(np.log2(end_size))
        start_bits = int(np.log2(start_size))

        M_vec = 1 << np.arange(start_bits, end_bits + 1)
        K_terms = OrderedDict()
        offset_terms = OrderedDict()
        K_step = None
        msb_terms = OrderedDict()
        for M in M_vec:
            fc = (1. / M) * fc_scale
            trans_bw = (2./ M) * tbw_scale
            print("trans_bw = {}".format(trans_bw))
            self.num_taps = M * self.taps_per_phase
            self.M = M
            filter_obj = fil_utils.LPFilter(M=M, P=M, pbr=self.pbr, sba=self.sba, num_taps=self.num_taps, fc=fc,
                                            num_iters=NUM_ITERS, fc_atten=SIX_DB, qvec=self.qvec,
                                            qvec_coef=self.qvec_coef, quick_gen=True, trans_bw=trans_bw, K=K_init, K_step=K_step,
                                            num_iters_min=100, freqz_pts=self.freqz_pts)

            K_terms[M] = filter_obj.K
            offset_terms[M] = filter_obj.offset
            taps = self.gen_float_taps(self.gen_2X, K_terms, offset_terms, M)
            msb_terms[M] = self.pfb_msb
            # use optimized parameter as the first guess on the next filter
            K_step = .01
            K_init = filter_obj.K
            self.gen_fixed_filter(taps)

        return K_terms, msb_terms, offset_terms

    def plot_filter(self, miny=-100, w_time=True, fft_size=16384):
        """
            Helper function that plots the PSD of the filter.
        """
        plot_title = "Channelizer Filter Impulse Response"
        limit = 4 * self.fc
        step = self.fc / 50.
        freq_vector = np.arange(-limit, limit, step)
        self.plot_psd(title=plot_title, pwr_pts=SIX_DB, fft_size=fft_size, miny=-100, freq_vector=freq_vector)

        plot_title = "Channelizer Filter Impulse Response Full"
        self.plot_psd(title=plot_title, pwr_pts=SIX_DB, fft_size=fft_size, miny=-180)

    @staticmethod
    def gen_cen_freqs(paths):

        half_step = 1. / paths
        full_step = half_step * 2
        num_steps = paths // 2 - 1

        init_list = [0]
        left_side = [-full_step - val * full_step for val in reversed(range(num_steps))]
        right_side = [full_step + val * full_step for val in range(num_steps + 1)]
        init_list = left_side + init_list + right_side

        return init_list

    @staticmethod
    def conv_bins_to_centers(paths, bins):
        centers = np.roll(np.fft.fftshift(Channelizer.gen_cen_freqs(paths)), 1)
        return [centers[bin] for bin in bins]

    @staticmethod
    def circ_shift(in_vec, paths):
        """
            Implements the circular shift routine of the Channelizer algorithm
        """
        shift_out = []
        for i, fil_arm in enumerate(in_vec):
            if i % 2:
                shift_out.append(np.roll(fil_arm, paths // 2))
            else:
                shift_out.append(fil_arm)

        return np.asarray(shift_out)

    @staticmethod
    def pf_run(sig_array, pf_bank, paths, rate=1):
        """
            Runs the input array through the polyphase filter bank.
        """
        fil_out = []
        offset = paths // rate
        for j in range(rate):
            for i, _ in enumerate(sig_array):
                # remember in channelizer samples are fed to last path first -- it is a decimating filter.
                index = i + j * offset
                fil_out.append(signal.upfirdn(pf_bank[index, :], sig_array[i, :]))


        return np.asarray(fil_out)

    @staticmethod
    def trunc_vec(input_vec, paths, gen_2X):
        mod_term = paths
        if gen_2X:
            mod_term = paths // 2
        trunc = len(input_vec) % mod_term
        if trunc > 0:
            input_vec = input_vec[:-trunc]

        return input_vec

    def ret_taps_fi(self):
        taps = self.gen_float_taps(self.gen_2X, self.K, self.M)
        ipdb.set_trace()

    def gen_tap_roms(self, path=None, file_prefix=None, roll_start=0, roll_offset=0, qvec_coef=(25, 24), qvec=(18, 17)):
        """
            Helper function that generates the coe files to be used with the PFB logic.
        """
        pfb_fil = copy.deepcopy(self.poly_fil_fi)
        # convert each column into ROM
        pfb_fil = pfb_fil.T
        qvec = (self.qvec_coef[0], 0)
        for idx, col in enumerate(pfb_fil):
            fi_obj = fp_utils.ret_dec_fi(col, qvec)
            if file_prefix is None:
                file_name = 'pfb_col_{}.coe'.format(idx)
            else:
                file_name = '{}_pfb_col_{}.coe'.format(file_prefix, idx)
            if path is not None:
                file_name = path + 'pfb_taps_{}/'.format(idx) + file_name

            fp_utils.coe_write(fi_obj, radix=16, file_name=file_name, filter_type=False)

    def gen_tap_file(self, file_name=None):
        """
            Helper function that generates a single file used for programming the internal ram
        """
        pfb_fil = copy.deepcopy(self.poly_fil_fi)
        pfb_fil = pfb_fil.T
        vec = np.array([])
        pad = np.array([0] * (self.Mmax - self.M))
        for col in pfb_fil:
            col_vec = np.concatenate((col, pad))
            vec = np.concatenate((vec, col_vec))

        print(len(vec))
        write_binary_file(vec, file_name, 'i', big_endian=True)

    def gen_mask_vec(self, percent_active=None, bin_values=[42, 43, 56]):

        if percent_active is not None:
            np.random.seed(10)
            num_bins = np.min((int(self.M * percent_active), self.max_masks))
            bin_values = np.random.choice(a=self.M, size=num_bins, replace=False)
            bin_values = np.sort(bin_values)
        # map this vector to 32 bit words  -- there are 64 words in 2048 bit vector for example.
        bit_vector = [0] * 65536
        num_words = int(np.ceil(self.M / 32))
        for value in bin_values:
            bit_vector[value] = 1

        words = np.reshape(bit_vector, (-1, 32))
        words = np.fliplr(words)
        words = words[:num_words, :]

        ret_val = np.atleast_1d(fp_utils.list_to_uint(words))
        return ret_val, bin_values

    def gen_mask_file(self, file_name=None, percent_active=None, bin_values=[42, 43, 56]):
        """
            Helper function that generates a single file used for programming the internal ram
        """
        vec, bin_values = self.gen_mask_vec(percent_active, bin_values)
        write_binary_file(np.array(vec), file_name, 'I', big_endian=True)
        return bin_values

    @staticmethod
    def gen_pf_bank(poly_fil, paths, rate):
        """
            Generates appropriate form of the polyphase filter bank to be used in the
            channelizer.
        """
        pf_bank = copy.copy(poly_fil)
        # modify pf_bank if gen_2X
        if rate == 2:
            pf_ret = []
            for i, pf_row in enumerate(pf_bank):
                if i < (paths // 2):
                    pf_ret.append(upsample(pf_row, 2, 0))
                else:
                    pf_ret.append(upsample(pf_row, 2, 1))

            return np.asarray(pf_ret)
        else:
            return np.asarray(pf_bank)

    def analysis_bank(self, input_vec, plot_out=False):
        """
            Function generates the analysis bank form of the channelizer.
        """
        # reshape input_vec

        input_vec = Channelizer.trunc_vec(input_vec, self.paths, self.gen_2X)
        if plot_out:
            plot_psd_helper(input_vec, fft_size=1024, title='Buffer Sig', miny=None, plot_time=False, markersize=None,
                            plot_on=True, savefig=True)


        sig_array = np.flipud(np.reshape(input_vec, (self.paths // self.rate, -1), order='F'))

        pf_bank = Channelizer.gen_pf_bank(self.poly_fil, self.paths, self.rate)
        num_plots = self.M
        if (plot_out):
            plt_sig = input_vec[:self.M * 10000]
            plt_array = np.reshape(plt_sig, (self.M // self.rate, -1), order='F')
            buff_array = []
            for j in range(10000):
                samp0 = plt_array[:, j]
                temp = np.concatenate((samp0, samp0))
                buff_array.extend(temp.tolist())


        fil_out = Channelizer.pf_run(sig_array, pf_bank, self.paths, self.rate)

        if (plot_out):
            plt_array = np.reshape(fil_out[:,:], (1, -1), order='F').flatten()
            plt_array = plt_array[:self.M * 100000]
            # for ii in range(num_plots):
            plot_psd_helper(plt_array, title='PFB Sig', miny=None, w_time=True, markersize=None,
                            plot_on=True, savefig=True)

        # now perform circular shifting if this is a 2X filter bank.
        shift_out = Channelizer.circ_shift(fil_out.transpose(), self.paths) if self.rate == 2 else fil_out.transpose()
        if (plot_out):
            shift_tp = shift_out.transpose()
            plt_array = np.reshape(shift_tp, (1, -1), order='F').flatten()
            plt_array = plt_array[:self.M * 100000]
            plot_psd_helper(plt_array, fft_size=1024, title='Circ Shift Sig', miny=None, w_time=True, markersize=None,
                            plot_on=True, savefig=True)

        # chan_out = np.fft.fftshift(np.fft.ifft(shift_out, axis=1), axes=1)
        chan_out = np.fft.ifft(shift_out, axis=1)

        if (plot_out):
            for ii in range(num_plots):
                fig, (ax0, ax1) = plt.subplots(2)
                ax0.plot(np.real(chan_out[:, ii]))
                ax1.plot(np.imag(chan_out[:, ii]))
                title = 'IFFT Output #{}'.format(ii)
                fig.canvas.manager.set_window_title(title)
                fig.savefig(title + '.png', figsize=(12, 10))

        return chan_out.transpose()

    def synthesis_bank(self, input_vec, plot_out=False):
        """
            Function generates the synthesis bank of the channelizer.
        """
        input_vec = Channelizer.trunc_vec(input_vec, self.paths, self.gen_2X)
        sig_array = np.reshape(input_vec, (self.paths, -1), order='F')
        pf_bank = Channelizer.gen_pf_bank(self.poly_fil, self.paths, self.rate)

        fft_out = np.fft.ifft(self.paths * sig_array, axis=0)

        shift_out = Channelizer.circ_shift(fft_out.transpose(), self.paths)
        fil_out = self.rate * Channelizer.pf_run(shift_out.transpose(), pf_bank, self.paths, 1)

        if self.rate == 2:
            offset = self.paths // 2
            for i in range(self.paths // 2):
                fil_out[i, :] = (fil_out[i, :] + fil_out[i + offset, :])

            fil_out = fil_out[:offset, :]

        return np.reshape(fil_out, (1, -1), order='F').flatten()

    def plot_phase_csum(self, pf_up=None):

        if pf_up is None:
            pf_up = self.gen_usample_pf()

        _, _, ref_phase = self.gen_freq_phase_profiles(pf_up)

        nfft = np.shape(pf_up)[1]
        freq1 = np.fft.fftshift(np.fft.fft(pf_up, nfft, 1))

        freq_sum = np.sum(freq1, 1)
        freq_abs = np.abs(freq_sum)
        phase_sum = np.unwrap(np.angle(freq_sum)) / (2 * np.pi)
        arg1 = phase_sum - ref_phase

        fig = plt.figure()
        ax = Axes3D(fig)
        # ax = fig.add_subplot(111, projection='3d')

        exp_vec = np.exp(1j * 2 * np.pi * arg1)
        x_vec = np.imag(freq_abs * exp_vec)
        y_vec = np.arange(-1, 1., 2. / len(freq_abs))
        z_vec = np.real(freq_abs * exp_vec)
        ax.plot(x_vec, y_vec, z_vec)  # , rstride=10, cstride=10)
        ax.set_xlabel(r'$\sf{Imag}')
        ax.set_ylabel(r'$\sf{Freq}')
        ax.set_zlabel(r'$\sf{Real}')
        ax.view_init(30, 10)
        ax.set_ylim(-1, 1.)
        ax.set_xlim(-1, 1)
        ax.set_zlim(-1, 1)
        title = 'Phase Rotators'
        fig.savefig(title+'.png', fig_size=(12, 10))

    def gen_usample_pf(self):

        coef_array = []
        # coef_array = self.filter.poly_fil
        # Upsample Filter Coefficients based on M.
        pf_up = []
        if self.gen_2X:
            b_temp = self.taps
            x_vec = np.arange(0, len(b_temp))
            x_vec2 = np.arange(0, len(b_temp), .5)
            b_temp = sp.interp(x_vec2, x_vec, b_temp) / 2.
            coef_array = self.gen_poly_partition(b_temp)
        else:
            coef_array = self.poly_fil

        for i, path_taps in enumerate(coef_array):
            # upsample path taps so that the paths show the prospective "paddle wheels"
            temp = upsample(path_taps, self.paths)
            # samples.
            pf_up.append(np.roll(temp, i))

        max_gain = np.max(np.sum(pf_up, 1))
        pf_up = [max_gain / np.sum(value) * value for value in pf_up]

        return pf_up

    def gen_freq_phase_profiles(self, pf_up=None):

        if pf_up is None:
            pf_up = self.gen_usample_pf()

        freq_path = []
        phase_path = []
        for taps_bb in pf_up:
            temp = np.fft.fftshift(np.fft.fft(taps_bb))
            freq_path.append(np.abs(temp))
            phase_path.append(np.unwrap(np.angle(temp)) / (2 * np.pi))

        ref_phase = phase_path[0]

        return (freq_path, phase_path, ref_phase)

    def gen_animation(self, fps=15, dpi_val=300, mpeg_file='test.mp4', sleep_time=.02):
        sel = 1
        ph_steps = 300
        inc = sel / float(ph_steps)
        num_frames = 10
        std_dev = np.sqrt(.02 / self.paths)
        mean = .5 / self.paths
        sig_bws = np.abs(std_dev * np.random.randn(self.paths) + mean)

        sig_bws = [value if value > .1 else .1 for value in sig_bws]
        cen_freqs = Channelizer.gen_cen_freqs(self.paths)
        mod_obj = QAM_Mod()
        sig = None
        for ii, (sig_bw, cen_freq) in enumerate(zip(sig_bws, cen_freqs)):  # cen_freqs:
            temp, _ = mod_obj.gen_frames(num_frames=num_frames, cen_freq=cen_freq, frame_space_mean=0, frame_space_std=0, sig_bw=sig_bw)
            idx = np.argmax(np.abs(temp))

            lidx = idx - 5000
            ridx = lidx + 20000

            if ii == 0:
                sig = temp[lidx:ridx]
            else:
                sig_temp = temp[lidx:ridx]
                sig[:len(sig_temp)] += sig_temp

        sig, _ = gen_utils.add_noise_pwr(30, sig)
        
        plot_psd_helper(sig)
        FFMpegWriter = manimation.writers['ffmpeg']  # ['ffmpeg']  avconv
        # metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        metadata = dict(artist='Matplotlib')

        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,metadata=None)

        pf_up = self.gen_usample_pf()
        fig = plt.figure()
        fig.set_tight_layout(False)

        # writer = MovieWriter(fig, frame_format= , fps=fps)
        ax = fig.add_subplot(221, projection='3d')
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax1 = fig.add_subplot(222, projection='3d')
        ax1.xaxis.labelpad = 10
        ax1.yaxis.labelpad = 10
        ax2 = fig.add_subplot(223, projection='polar')
        ax3 = fig.add_subplot(224)
        fig.subplots_adjust(left=.08, bottom=.13, top=.95, right=.96, hspace=.4, wspace=.2)
        y_vec = np.arange(-1, 1., 2. / np.shape(pf_up)[1])

        phase_vec = np.arange(0, sel + 5 * inc, inc)
        # pad the end with the last phase for 30 frames
        phase_vec = np.concatenate((phase_vec, np.array([phase_vec[-1]]*30)))

        (_, sig_psd) = gen_psd(sig, fft_size=len(y_vec))
        with writer.saving(fig, mpeg_file, dpi_val):
            for phase in phase_vec:
                if phase > 1:
                    m = 1
                else:
                    m = phase
                indices = np.arange(0, self.M)
                rot = np.exp(1j * 2 * np.pi * (indices / float(self.M)) * m)
                pf_up3 = [rot_value * row for (rot_value, row) in zip(rot, pf_up)]
                (freq_path, phase_path, ref_phase) = self.gen_freq_phase_profiles(pf_up3)
                x_sum = 0
                z_sum = 0
                for i, (f_path, p_path) in enumerate(zip(freq_path, phase_path)):
                    arg1 = p_path - ref_phase

                    # x term is the imaginary component frequency Response of path ii
                    # rotated by the phase response of path ii
                    exp_vec = np.exp(1j * 2 * np.pi * arg1)
                    x_vec = f_path * np.imag(exp_vec)
                    z_vec = f_path * np.real(exp_vec)
                    x_sum += x_vec
                    z_sum += z_vec
                    if i == 0:
                        ax.set_xlabel(r'$\sf{Imag}$')
                        ax.set_ylabel(r'$\sf{Freq}$')
                        ax.set_zlabel(r'$\sf{Real}$')
                        ax.view_init(30, 10)
                        ax.set_ylim(-1, 1.)
                        ax.set_xlim(-1, 1)
                        ax.set_zlim(-1, 1)
                        ax.set_title(r'$\sf{Phase\ Arms}$')
                        ax.plot(np.array([0, 0]), np.array([0, 0]), np.array([-1.2, 1.2]), color='k', linewidth=.5)
                        ax.plot(np.array([0, 0]), np.array([-1.2, 1.2]), np.array([0, 0]), color='k', linewidth=.5)
                        ax.plot(np.array([-1.2, 1.2]), np.array([0, 0]), np.array([0, 0]), color='k', linewidth=.5)

                    # ax.plot_wireframe(x_vec, y_vec, z_vec)  # , rstride=10, cstride=10)
                    ax.plot(x_vec, y_vec, z_vec, linewidth=.9)  # , rstride=10, cstride=10)

                x_sum = x_sum / self.paths
                z_sum = z_sum / self.paths

                ax1.set_xlabel(r'$\sf{Imag}$')
                ax1.set_ylabel(r'$\sf{Freq}$')
                ax1.set_zlabel(r'$\sf{Real}$')
                ax1.view_init(30, 10)
                ax1.set_ylim(-1, 1.)
                ax1.set_xlim(-1, 1)
                ax1.set_zlim(-1, 1)
                ax1.set_title(r'$\sf{Phase Coherent Sum}$')
                ax1.plot([0, 0], [0, 0], [-1.2, 1.2], color='k', linewidth=.5)
                ax1.plot([0, 0], [-1.2, 1.2], [0, 0], color='k', linewidth=.5)
                ax1.plot([-1.2, 1.2], [0, 0], [0, 0], color='k', linewidth=.5)
                ax1.plot(x_sum, y_vec, z_sum, linewidth=.9)

                rot = [np.exp(1j * 2 * np.pi * (ii / float(self.paths)) * m) for ii in range(self.paths)]

                compass(np.real(rot), np.imag(rot), ax2)
                ax2.set_xlabel(r'$\sf{Phase\ rotator\ progression}')

                fil = z_sum + 1j * x_sum
                fil_log = 20 * np.log10(np.abs(fil))
                out_log = sig_psd + fil_log
                plot_psd(ax3, y_vec, out_log, miny=-85, maxy=15, titlesize=12, labelsize=10, xlabel=df_str)

                # fig.subplots_adjust(top=.95)   # tight_layout(h_pad=.5)
                writer.grab_frame()
                time.sleep(sleep_time)
                ax.clear()
                ax1.clear()
                ax2.clear()
                ax3.clear()

        plt.close(fig)

    def gen_properties(self, plot_on=True):
        """
            Generates plots related to the analysis bank of the designed filter.
        """
        self.paths = self.M
        # Upsample Filter Coefficients based on M.
        pf_up = self.gen_usample_pf()

        # now insert the extra delay if this is a 2X implementation.
        b_nc = np.fft.fftshift(self.taps)
        bb1 = np.reshape(b_nc, (self.paths, -1), order='F')
        nfft = len(self.taps)

        fig, ax = plt.subplots()
        title = r'$\sf{Polyphase\ Filter\ Phase\ Profiles}$'

        x_vec = np.arange(-1, 1., (2. / nfft)) * self.paths
        phs_sv = []
        for i in range(self.paths):
            fft_out = np.fft.fftshift(np.fft.fft(pf_up[i], nfft))
            phs = np.unwrap(np.angle(fft_out))
            temp = phs[nfft // 2]
            phs_sv.append(temp)
            ax.plot(x_vec, phs - temp)

        ax.set_xlabel('Nyquist Zones')
        ax.set_ylabel('Phase (radians)')
        ax.set_title(title)
        fig.canvas.manager.set_window_title(title)
        fig.savefig(title, figsize=(12, 10))

        title = r'$\sf{Reference\ Partition}$'
        fig, ax = plt.subplots()
        ax.stem(bb1[0])
        ax.set_title(r'$\sf{Polyphase Filter\ --\ Reference\ Partition}$')
        fig.canvas.manager.set_window_title(title)
        fig.savefig(title, figsize=(12, 10))

        if plot_on:
            (freq_path, phase_path, ref_phase) = self.gen_freq_phase_profiles(pf_up)

            ax = []
            fig = plt.figure()
            for i, (f_path, p_path) in enumerate(zip(freq_path, phase_path)):
                arg1 = p_path - ref_phase
                # x term is the imaginary component frequency Response of path ii
                # rotated by the phase response of path ii
                exp_vec = np.exp(1j * 2 * np.pi * arg1)
                x_vec = np.imag(f_path * exp_vec)
                y_vec = np.arange(-1, 1., 2. / len(f_path))
                z_vec = np.real(f_path * exp_vec)
                ax.append(fig.add_subplot(2, self.paths / 2, i + 1, projection='3d'))
                ax[i].plot_wireframe(x_vec, y_vec, z_vec)  # , rstride=10, cstride=10)
                ax[i].set_xlabel('Imag')
                ax[i].set_ylabel('Freq')
                ax[i].set_zlabel('Real')
                ax[i].view_init(30, 10)
                ax[i].set_ylim(-1, 1.)
                ax[i].set_xlim(-1, 1)
                ax[i].set_zlim(-1, 1)

            fig.savefig('Properties.png', figsize=(12, 10))


def gen_test_sig(fft_size, file_name=None, bursty=False, bins=[0, 3, 4, 7], sig_bws = [.1, .10, .05, .125], amps = [.2, .5, .7, .3], roll=[1000, 3000, 4000, 5000]):

    cen_freqs = Channelizer.conv_bins_to_centers(fft_size, bins)    

    # bursty
    if bursty:
        num_frames = 10
        packet_size = 100
        frame_space_mean = 20000
    else:
        num_frames = 1
        packet_size = 10000
        frame_space_mean = 0

    sig_list = []
    if file_name is None:
        file_name = SIM_PATH + 'sig_store_test8.bin'
    mod_obj = QAM_Mod(frame_mod='qam16', xcode_shift=2, ycode_shift=2, packet_size=packet_size)

    for cen_freq, sig_bw, amp, shift in zip(cen_freqs, sig_bws, amps, roll):
        temp, _ = mod_obj.gen_frames(num_frames=num_frames, cen_freq=cen_freq, frame_space_mean=frame_space_mean,
                                     frame_space_std=0, sig_bw=sig_bw, snr=200)

        if not bursty:
            idx = np.where(np.abs(temp) > .5)[0][0]
            temp = np.asarray(temp[idx:idx + 1000000])
        temp *= (amp * .1)
        temp = np.roll(temp, shift)
        sig_list.append(temp)

    min_length = min([len(temp) for temp in sig_list])
    sig = 0
    for temp in sig_list:
        sig += temp[:min_length]
    sig, _ = add_noise_pwr(80, sig)
    sig_fi = fp_utils.ret_fi(sig, qvec=(16, 15), overflow='saturate')

    plot_psd_helper(sig_fi.vec, title='Input Spectrum', savefig=True, plot_time=True, dpi=100)
    write_complex_samples(sig_fi.vec, file_name, False, 'h', big_endian=True)
    plt.show()


def gen_corr_table(num_bits=6):

    qvec = (num_bits, num_bits)
    vec = fp_utils.comp_range_vec(qvec)
    new_vec = 2 ** -vec
    path = IP_PATH
    new_vec_fi = fp_utils.ret_dec_fi(new_vec, qvec=(16, 15), signed=0)
    print(vgen.gen_rom(path, new_vec_fi, rom_type='sp', rom_style='distributed', prefix='exp_shift_'))

    return new_vec_fi


def gen_samp_delay_coe(M):

    qvec = (36, 0)

    vec = np.array([0] * M)
    fi_obj = fp_utils.ret_dec_fi(vec, qvec)

    file_name = IP_PATH + '/sample_delay/sample_delay.coe'
    fp_utils.coe_write(fi_obj, radix=16, file_name=file_name, filter_type=False)

    file_name = IP_PATH + '/sample_ram/sample_ram.coe'

    ridx = 3 * M + 1
    vec = np.arange(1, ridx)  # np.array([0] * M * 3)
    fi_obj = fp_utils.ret_dec_fi(vec, qvec)
    fp_utils.coe_write(fi_obj, radix=16, file_name=file_name, filter_type=False)

    vec = np.array([0] * M)
    qvec = (36, 0)
    fi_obj = fp_utils.ret_dec_fi(vec, qvec)

    file_name = IP_PATH + '/circ_buff_ram/circ_buff_ram.coe'
    fp_utils.coe_write(fi_obj, radix=16, file_name=file_name, filter_type=False)

    file_name = IP_PATH + '/exp_averager_filter/exp_fil.coe'
    fil_vec = [1] * 64
    fi_obj = fp_utils.ret_dec_fi(fil_vec, (2, 0))
    fp_utils.coe_write(fi_obj, radix=16, file_name=file_name, filter_type=True)


def gen_taps(M, Mmax=512, taps_per_phase=TAPS_PER_PHASE, dsp48e2=False):

    qvec_coef = ret_qcoef(dsp48e2)
    chan = Channelizer(M=M, Mmax=Mmax, gen_2X=GEN_2X, taps_per_phase=taps_per_phase,
                       desired_msb=PFB_MSB, qvec=QVEC, qvec_coef=qvec_coef, fc_scale=FC_SCALE)
    print("Filter MSB = {}".format(chan.fil_msb))
    path = SIM_PATH
    file_name = path + 'M_{}_taps.bin'.format(M)
    print(file_name)
    chan.gen_tap_file(file_name)


def populate_fil_table(start_size=8, end_size=2048, taps_per_phase=TAPS_PER_PHASE, fc_scale=FC_SCALE, 
                       gen_2X=GEN_2X, tbw_scale=TBW_SCALE, freqz_pts=10_000, qvec=QVEC, qvec_coef=QVEC_COEF):

    # K_init = 20. if gen_2X else 40.
    K_init = 40.

    chan = Channelizer(M=8, gen_2X=gen_2X, taps_per_phase=taps_per_phase, qvec=qvec, qvec_coef=qvec_coef,
                       fc_scale=fc_scale, tbw_scale=tbw_scale, freqz_pts=freqz_pts)

    K_terms, msb_terms, offset_terms = chan.gen_fil_params(start_size, end_size, fc_scale=fc_scale, K_init=K_init, 
                                                           tbw_scale=tbw_scale)

    print("K_terms = {}".format(K_terms))
    print("msb_terms = {}".format(msb_terms))
    print("offset_terms = {}".format(offset_terms))

    return K_terms, msb_terms, offset_terms

def gen_animation():
    chan_obj = Channelizer(M=4, taps_per_phase=TAPS_PER_PHASE, gen_2X=False, qvec=QVEC, fc_scale=FC_SCALE)
    chan_obj.gen_animation()

def find_best_terms(gen_2X=True, qvec=QVEC):
    # M = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # M = 2 ** np.arange(2, 7)
    M = 64
    chan_obj = Channelizer(M=M, taps_per_phase=TAPS_PER_PHASE, gen_2X=GEN_2X, qvec=QVEC,
                           fc_scale=FC_SCALE, tbw_scale=TBW_SCALE, K_terms=K_default, )

    K_terms, msb_terms, offset_terms = chan_obj.gen_fil_params(8, 2048)

    print("K_terms = {}".format(K_terms))
    print("msb_terms = {}".format(msb_terms))
    print("offset_terms = {}".format(offset_terms))

def gen_input_buffer(Mmax=512, path=IP_PATH, gen_2X=False):

    cnt_width = 16  #  input buffers are fixed code -- always use 16 bit counters
    print("==========================")
    print(" input buffer")
    print("")
    ram_out = vgen.gen_ram(path, ram_type='dp', memory_type='read_first', ram_style='block')
    print(ram_out)
    cnt_in, in_fifo = vgen.gen_aligned_cnt(path, cnt_width=cnt_width, tuser_width=0, tlast=False, start_sig=False, dwn_cnt=True)
    print(cnt_in)
    cnt_out, out_fifo = vgen.gen_aligned_cnt(path, cnt_width=cnt_width, tuser_width=0, tlast=False, start_sig=True, use_af=True,
                                   almost_full_thresh=16, fifo_addr_width=5)
    print(cnt_out)
    print("==========================")
    print("")
    name = "input_buffer_1x" if gen_2X is False else "input_buffer"
    return name, cnt_in, cnt_out, in_fifo, out_fifo



def gen_output_buffer(Mmax=512, path=IP_PATH):

    print("==========================")
    print(" output buffer")
    print("")
    cnt_width = 16  #int(np.ceil(np.log2(Mmax - 1)))
    ram_out = vgen.gen_ram(path, ram_type='dp', memory_type='read_first', ram_style='block')
    print(ram_out)
    # cnt_in = vgen.gen_aligned_cnt(path, cnt_width=cnt_width, tuser_width=0, tlast=True, start_sig=False, dwn_cnt=False)
    # vgen.gen_aligned_cnt(path, cnt_width=16, tuser_width=0, tlast=False, incr=1, tot_latency=None, start_sig=False, cycle=False, upper_cnt=False, prefix='', dwn_cnt=False, load=False, dport=True, startup=True, almost_full_thresh=None)
    # print(cnt_in)
    cnt_out = vgen.gen_aligned_cnt(path, cnt_width=cnt_width, tuser_width=0, tlast=False, start_sig=True, use_af=True,
                                   almost_full_thresh=16, fifo_addr_width=5)
    print(cnt_out)
    _, dsp_name = vgenx.gen_dsp48E1(path, 'output_add', opcode='A+D', a_width=QVEC[0], d_width=QVEC[0], dreg=1, areg=1, creg=2, mreg=1, breg=0, preg=1, rnd=True, p_msb=QVEC[0], p_lsb=1)
    print(dsp_name)
    print("==========================")
    print("")


def gen_one_hot(Mmax=512, path=IP_PATH):
    print("==========================")
    print(" one hot encoder")
    print("")
    input_width = int(np.ceil(np.log2(Mmax-1)))
    one_hot_out = vgen.gen_one_hot(input_width, file_path=path)
    print(one_hot_out)
    print("==========================")
    print("")

def gen_downselect(Mmax=512, path=IP_PATH):
    print("==========================")
    print(" one hot encoder")
    print("")
    tuser_bits = calc_fft_tuser_width(Mmax)
    print("tuser_bits = {}".format(tuser_bits))
    downselect, mux_out = adv_filter.gen_down_select(path, name='downselect', num_channels=Mmax, tuser_width=tuser_bits)
    print(downselect)
    print("==========================")
    print("")
    return downselect, mux_out

def gen_mux(Mmax=512, path=IP_PATH):
    print("==========================")
    print(" pipelined mux")
    print("")
    input_width = Mmax
    mux_out = vgen.gen_pipe_mux(path, input_width, 1, mux_bits=3, one_hot=False, one_hot_out=False)
    print(mux_out)
    print("==========================")
    print("")
    return mux_out

def gen_pfb(chan_obj, path=IP_PATH, fs=6.4E6, dsp48e2=False, fc_scale=FC_SCALE):  # Mmax=512, pfb_msb=40, M=512, taps=None, gen_2X=GEN_2X):
    """
        Generates the logic for the Polyphase Filter bank
    """

    # path = IP_PATH
    print("==========================")
    print(" pfb filter")
    print("")

    print("K terms = {}".format(chan_obj.K))
    print("fc_scale = {}".format(fc_scale))
    pfb_fil = chan_obj.poly_fil_fi
    pfb_reshape = pfb_fil.T.flatten()
    qvec_coef = ret_qcoef(dsp48e2) 
    qvec_int = (qvec_coef[0], 0)
    taps_fi = fp_utils.ret_dec_fi(pfb_reshape, qvec_int)

    mid_pt = len(taps_fi.vec) // 2
    print("Taps binary")
    [print(value) for value in taps_fi.bin[mid_pt-10:mid_pt+10]]

    fft_size=8192
    step = 1 / (32768 * 8)
    freq_vector = np.arange(-10. / chan_obj.M, 10. / chan_obj.M, step)
    omega_scale = (fs / 2) / chan_obj.M
    xlabel = r'$\sf{kHz}$'
    chan_obj.plot_psd(fft_size=fft_size, taps=None, freq_vector=freq_vector, title=r'$\sf{Filter\ Prototype}$', savefig=True,
                      pwr_pts=SIX_DB, miny=-120, omega_scale=omega_scale, xlabel=xlabel)

    pfb_out = adv_filter.gen_pfb(path, chan_obj.Mmax, taps_fi, input_width=chan_obj.qvec[0], output_width=chan_obj.qvec[0],
                                 taps_per_phase=chan_obj.taps_per_phase, pfb_msb=chan_obj.pfb_msb, 
                                 tlast=True, tuser_width=0, ram_style='block', prefix='', gen_2X=chan_obj.gen_2X, dsp48e2=dsp48e2)

    print(pfb_out)
    print("==========================")
    return pfb_out

def gen_circ_buffer(Mmax=512, path=IP_PATH):
    print("======================")
    print("circular buffer")
    print("")
    ram_out = vgen.gen_ram(path, ram_type='dp', memory_type='read_first', ram_style='block')
    print(ram_out)
    fifo_out = vgen.gen_axi_fifo(path, tuser_width=0, tlast=True, almost_full=True, ram_style='distributed')
    print(fifo_out)
    print("======================")
    print("")
    return fifo_out

def gen_final_cnt(path=IP_PATH):
    print("======================")
    print("Final Count")
    print("")
    final_cnt, _ = vgen.gen_aligned_cnt(path, cnt_width=16, tuser_width=24, tlast=True)
    print(final_cnt)
    print("======================")
    print("")
    return final_cnt


def gen_exp_shifter(chan_obj, avg_len=16, path=IP_PATH):
    print("=================================")
    print("exp shifter")
    print("")
    # generate correction ROM
    table_bits = fp_utils.ret_bits_comb(avg_len)
    frac_bits = int(np.ceil(np.log2(avg_len)))
    fft_shift_bits = 5
    qvec_in = (fft_shift_bits, 0)
    qvec_out = (fft_shift_bits + frac_bits, frac_bits)
    cic_obj = fil_utils.CICDecFil(M=avg_len, N=1, qvec_in=qvec_in, qvec_out=qvec_out)
    _, cic_name, slicer_name, cic_fifo, comb_name, comb_fifo = vfilter.gen_cic_top(path, cic_obj, count_val=0, prefix='', tuser_width=0)
    # fil_msb = fft_shift_bits + table_bits - 1
    # fi_obj = fp_utils.ret_dec_fi([1.] * avg_len, qvec=(25, 0), overflow='wrap', signed=1)

    # generate fifo
    _, fifo_out = vgen.gen_axi_fifo(path, tuser_width=24, tlast=True, almost_full=True, ram_style='distributed')    
    print(fifo_out)
    print("================================")
    print("")

    _, exp_out = adv_filter.gen_exp_shift_rtl(path, chan_obj, cic_obj)
    print(exp_out)
    print("================================")
    print("")
    return exp_out, cic_name, cic_fifo, slicer_name, fifo_out, comb_name, comb_fifo

def gen_tones(M=512, lidx=30, ridx=31, offset=0, path=SIM_PATH):

    scale = np.max(np.abs((lidx, ridx)))
    tone_vec = np.arange(lidx, ridx, 1) / float(M / 2) + offset
    phase_vec = np.arange(lidx, ridx, 1) / (scale * np.pi)
    num_samps = 8192 * 256
    tones = [gen_comp_tone(num_samps, tone_value, phase_value) for (tone_value, phase_value) in zip(tone_vec, phase_vec)]
    sig = np.sum(tones, 0)
    sig = sig / (2. * np.max(np.abs(sig)))
    sig *= .5

    sig_fi = fp_utils.ret_fi(sig, qvec=(16, 15), overflow='saturate')

    plot_psd_helper(sig_fi.vec[:900], title='tone truth', miny=-100, savefig=True, plot_time=True, path=path)
    write_complex_samples(sig_fi.vec, path + 'sig_tones_{}.bin'.format(M), False, 'h', big_endian=True)

def gen_tones_vec(tone_vec, M=512, offset=0., path=SIM_PATH):

    omega = np.roll(gen_freq_vec(M)['w'], -M // 2)
    offset = [offset] * len(tone_vec) if not isinstance(offset, Iterable) else offset
    tones = np.array([omega[tone_idx] + off_value for tone_idx, off_value in zip(tone_vec, offset)])

    scale = np.max(tone_vec)
    phase_vec = np.array(tone_vec) / (scale * np.pi)
    num_samps = 8192 * 128
    tones = [gen_comp_tone(num_samps, tone_value, phase_value) for (tone_value, phase_value) in zip(tones, phase_vec)]
    sig = np.sum(tones, 0)
    sig = sig / (2. * np.max(np.abs(sig)))

    sig_fi = fp_utils.ret_fi(sig, qvec=(16, 15), overflow='saturate')

    # plot_psd_helper(sig_fi.vec[:900], title='tone truth', miny=-100, savefig=True, plot_time=True, path=path)
    write_complex_samples(sig_fi.vec, path + 'sig_tones_{}.bin'.format(M), False, 'h', big_endian=True)

def gen_count(M=512):
    path = SIM_PATH

    num_samps = 8192 * 256
    count = np.arange(0, num_samps)
    count = count % M
    count = count + 1j * 0

    sig_fi = fp_utils.ret_fi(count, qvec=(16, 0), overflow='saturate')
    write_complex_samples(sig_fi.vec, path + 'sig_count_{}.bin'.format(M), False, 'h', big_endian=True)

def process_pfb_out(file_name, row_offset=128):
    """
        Helper function that ingests data from RTL simulation and plots the output of the PFB of the channelizer.
    """
    samps = read_binary_file(file_name, format_str='Q', big_endian=True)
    print(len(samps))
    if type(samps) is int:
        print('File does not exist')
        return -1

    if len(samps) == 0:
        print("Not enough samples in File")
        return -1

    mask_i = np.uint64(((1 << 16) - 1) << 16)
    mask_q = np.uint64((1 << 16) - 1)
    mask_bin_num = np.uint64(((1 << 24) - 1) << 32)

    i_sig = [int(samp & mask_i) >> 16 for samp in samps]
    q_sig = [samp & mask_q for samp in samps]
    fft_bin_sig = [int(samp & mask_bin_num) >> 32 for samp in samps]

    offset = np.where(np.array(fft_bin_sig) == 0)[0][0]

    store_sig = np.array(i_sig[offset:] )+ 1j * np.array(q_sig[offset:])

    write_complex_samples(store_sig, './raw_pfb_out.bin', q_first=False)

    i_sig = fp_utils.uint_to_fp(i_sig[offset:], qvec=(16, 15), signed=1, overflow='wrap')
    q_sig = fp_utils.uint_to_fp(q_sig[offset:], qvec=(16, 15), signed=1, overflow='wrap')
    fft_bin_sig = fft_bin_sig[offset:]

    M = np.max(fft_bin_sig) + 1
    print("M = {}".format(M))

    trunc = len(i_sig) % M
    if trunc:
        i_sig = i_sig.float[:-trunc]
        q_sig = q_sig.float[:-trunc]
        fft_bin_sig = fft_bin_sig[:-trunc]
    else:
        i_sig = i_sig.float
        q_sig = q_sig.float


    comp_sig = i_sig + 1j * q_sig
    plot_psd_helper(comp_sig, w_time=True, savefig=True, title='Interlace PFB', miny=-80)
    comp_rsh = np.reshape(comp_sig, (M, -1), 'F')
    comp_rsh = np.fft.ifft(comp_rsh, axis=0)

    resps = []
    wvecs = []
    time_sigs = []
    print(np.shape(comp_rsh))
    for ii, row in enumerate(comp_rsh):
        if row_offset < len(row):
            row = row[row_offset:]
        print(np.max(np.abs(row)))

        wvec, psd = gen_psd(row, fft_size=256, window='blackmanharris')
        resps.append(psd)
        wvecs.append(wvec)
        time_sigs.append(row)
        lg_idx = np.argmax(np.abs(row))
        real_value = np.real(row[lg_idx])
        imag_value = np.imag(row[lg_idx])
        res_value = np.max(psd)
        print("{} : Largest value = {}, i{} - resp = {} db".format(ii, real_value, imag_value, res_value))

def process_inbuff(file_name):
    """
        Helper function that ingests data from RTL simulation and plots the output of the input buffer of the channelizer.
    """
    samps = read_binary_file(file_name, format_str='Q', big_endian=True)
    print(len(samps))
    if type(samps) is int:
        print('File does not exist')
        return -1

    if len(samps) == 0:
        print("Not enough samples in File")
        return -1

    mask_i = np.uint64(((1 << 16) - 1) << 16)
    mask_q = np.uint64((1 << 16) - 1)
    mask_bin_num = np.uint64(((1 << 24) - 1) << 32)

    i_sig = [int(samp & mask_i) >> 16 for samp in samps]
    q_sig = [samp & mask_q for samp in samps]
    fft_bin_sig = [int(samp & mask_bin_num) >> 32 for samp in samps]

    offset = np.where(np.array(fft_bin_sig) == 0)[0][0]
    print(offset)

    store_sig = np.array(i_sig[offset:] )+ 1j * np.array(q_sig[offset:])
    write_complex_samples(store_sig, './raw_inputbuffer.bin', q_first=False)

    i_sig = fp_utils.uint_to_fp(i_sig[offset:], qvec=(16, 15), signed=1, overflow='wrap')
    q_sig = fp_utils.uint_to_fp(q_sig[offset:], qvec=(16, 15), signed=1, overflow='wrap')
    fft_bin_sig = fft_bin_sig[offset:]

    M = np.max(fft_bin_sig) + 1
    print("M = {}".format(M))

    trunc = len(i_sig) % M
    if trunc:
        i_sig = i_sig.float[:-trunc]
        q_sig = q_sig.float[:-trunc]
        fft_bin_sig = fft_bin_sig[:-trunc]
    else:
        i_sig = i_sig.float
        q_sig = q_sig.float


    comp_sig = i_sig + 1j * q_sig
    plot_psd_helper(comp_sig, w_time=True, savefig=True, title='Input Buffer', miny=-80)

def process_chan_out(file_name, iq_offset=10*TAPS_PER_PHASE, Mmax=64, plot_on=False, dpi=100):
    """
        Helper function that ingests and plots the output of the channelizer from an RTL simulation.
    """
    samps = read_binary_file(file_name, format_str='Q', big_endian=True)
    tuser_bits = calc_fft_tuser_width(Mmax)
    print(len(samps))
    if type(samps) is int:
        print('File does not exist')
        return -1

    if len(samps) == 0:
        print("Not enough samples in File")
        return -1

    mask_i = np.uint64(((1 << 16) - 1) << 16)
    mask_q = np.uint64((1 << 16) - 1)
    mask_tuser = np.uint64(((1 << tuser_bits) - 1) << 32)
    mask_fft_bin = int(((1 << int(np.log2(Mmax))) -1))
    mask_tlast = np.uint64(1 << 32 + tuser_bits)

    i_sig = [(int(samp & mask_i) >> 16) for samp in samps]
    q_sig = [(samp & mask_q)  for samp in samps]
    tuser_sig = [int(samp & mask_tuser) >> 32 for samp in samps]
    fft_bin_sig = [int(samp & mask_fft_bin) for samp in tuser_sig]
    tlast_sig = [int(samp & mask_tlast) >> (32 + tuser_bits) for samp in samps]

    bin_list = np.unique(fft_bin_sig)
    offset = np.where(np.array(fft_bin_sig) == np.min(bin_list))[0][0]

    M = np.max(bin_list) + 1
    print("M = {}".format(M))

    i_sig = fp_utils.uint_to_fp(i_sig[offset:], qvec=(16, 15), signed=1, overflow='wrap')
    q_sig = fp_utils.uint_to_fp(q_sig[offset:], qvec=(16, 15), signed=1, overflow='wrap')
    tuser_sig = tuser_sig[offset:]
    tlast_sig = tlast_sig[offset:]
    fft_bin_sig = fft_bin_sig[offset:]

    iq_sig = np.array(i_sig.float[offset:] )+ 1j * np.array(q_sig.float[offset:])
    # partition streams into separate lists based on fft_bin_sig
    samp_lists = []
    for value in bin_list:
        indices = np.where(np.array(fft_bin_sig) == value)[0]
        iq_temp = [iq_sig[index] for index in indices]
        # print(len(iq_temp))
        samp_lists.append(iq_temp)

    resps = []
    wvecs = []
    time_sigs = []
    for iq_vec, bin_num in zip(samp_lists, bin_list):
        if iq_offset < len(iq_vec):
            iq_vec = iq_vec[iq_offset:]
        # print(np.max(np.abs(iq_vec)))
        wvec, psd = gen_psd(iq_vec, fft_size=1024, window='blackmanharris')
        resps.append(psd)
        wvecs.append(wvec)
        time_sigs.append(iq_vec)
        lg_idx = np.argmax(np.abs(iq_vec))
        real_value = np.real(iq_vec[lg_idx])
        imag_value = np.imag(iq_vec[lg_idx])
        res_value = np.max(psd)
        print("Bin {} \t: Largest value = {:.4f}, i{:.4f} - resp = {:.4f} db".format(bin_num, real_value, imag_value, res_value))

    if plot_on is False:
        mpl.use('Agg')
    # sig_list = np.arange(0, M).tolist()
    # sig_list = np.arange(35, 51).tolist()  #[41, 42, 43, 44, 45, 46]
    for j, bin_num in enumerate(bin_list):
        plot_psd_helper((wvecs[j], resps[j]), title=r'$\sf{{PSD\ Overlay\ {}}}$'.format(bin_num), plot_time=True, miny=-150, maxy=20.,
                        time_sig=time_sigs[j], markersize=None, plot_on=plot_on, savefig=True, ytime_min=-1., ytime_max=1., dpi=dpi)
        if plot_on is False:
            plt.close('all')


def process_synth_out(file_name, row_offset=600):

    samps = read_binary_file(file_name, format_str='I', big_endian=True)
    mod_name = gen_utils.ret_module_name(file_name)
    mod_name = mod_name.replace("_", " ")

    if type(samps) is int:
        print('File does not exist')
        return -1

    if len(samps) == 0:
        print("Not enough samples in File")
        return -1

    mask_i = np.uint32(((2 ** 16) - 1) << 16)
    mask_q = np.uint32((2 ** 16) - 1)

    i_sig = [int((samp & mask_i) / (1 << 16)) for samp in samps]
    i_sig = fp_utils.uint_to_fp(i_sig, qvec=(16, 15), signed=1, overflow='wrap')
    q_sig = [samp & mask_q for samp in samps]
    q_sig = fp_utils.uint_to_fp(q_sig, qvec=(16, 15), signed=1, overflow='wrap')

    i_sig = i_sig.float
    q_sig = q_sig.float

    comp_sig = i_sig + 1j * q_sig
    if len(comp_sig) < 2000:
        print("Not enough Synthesis Data : Data is {} samples".format(len(comp_sig)))
        return 1
    comp_sig = comp_sig[1000:]

    # plot_psd_helper(comp_sig, title='Synthesizer PSD Output - {}'.format(mod_name), w_time=True, miny=-100, maxy=None, plot_on=False, savefig=True)
    plot_psd_helper(comp_sig, title='Synthesizer PSD Output', w_time=True, miny=-100, maxy=None, plot_on=False, savefig=True)
    print("Synthesis Output Produced")


def gen_mask_files(M_list, percent_active=None, values=[42, 43, 44, 45, 46], path=SIM_PATH):
    bin_list = []
    for M in M_list:
        chan_obj = Channelizer(M=M, taps_per_phase=TAPS_PER_PHASE, gen_2X=GEN_2X)
        file_name = path + 'M_{}_mask.bin'.format(M)
        bin_values = chan_obj.gen_mask_file(file_name, percent_active, values)
        bin_list.append(bin_values)

    return bin_list

def gen_tap_plots(M_list):

    for M in M_list:
        chan_obj = Channelizer(M=M, taps_per_phase=TAPS_PER_PHASE, gen_2X=GEN_2X, K=K_default, 
                               desired_msb=DESIRED_MSB, fc_scale=FC_SCALE)
        # chan_obj.plot_psd_single(savefig=True)
        gen_taps(M, M_MAX, taps_per_phase=TAPS_PER_PHASE)
        print(chan_obj.pfb_msb)
        tap_title = 'taps psd M {}'.format(M)
        # get adjacent bins and plot suppression value.
        freq_pts = [- 1.25 / M, 1.25 / M, -1. / M, 1. / M]
        fft_size=16384
        if M > 256:
            fft_size = 16384
        chan_obj.plot_psd(fft_size=fft_size, taps=None, freq_vector=None, title=tap_title, savefig=True, pwr_pts=SIX_DB, freq_pts=freq_pts, miny=-120)

def plot_input_file(file_name):
    samps = read_binary_file(file_name, format_str='I', big_endian=True, num_samps=50000)
    # write_binary_file(samps, file_name, format_str='h', append=False, big_endian=True)
    mod_name = gen_utils.ret_module_name(file_name)
    mod_name = mod_name.replace("_", " ")
    if type(samps) is int:
        print('File does not exist')
        return -1

    if len(samps) == 0:
        print("Not enough samples in File")
        return -1

    mask_i = np.uint32(((2 ** 16) - 1) << 16)
    mask_q = np.uint32((2 ** 16) - 1)

    q_sig = [int((samp & mask_i) / (1 << 16)) for samp in samps]
    q_sig = fp_utils.uint_to_fp(q_sig, qvec=(16, 15), signed=1, overflow='wrap')
    i_sig = [samp & mask_q for samp in samps]
    i_sig = fp_utils.uint_to_fp(i_sig, qvec=(16, 15), signed=1, overflow='wrap')

    i_sig = i_sig.float
    q_sig = q_sig.float

    comp_sig = i_sig + 1j * q_sig

    plot_psd_helper(comp_sig, title='Stimulus PSD', w_time=True, miny=-100, maxy=None, plot_on=False, savefig=True, fft_size=2048)
    print("Input stimulus plotted")

def gen_logic(chan_obj, path=IP_PATH, avg_len=256, fs=6.4E6, gen_2X=False):
    """
        Helper function that generate RTL logic 
    """
    sample_fi = fp_utils.ret_dec_fi(0, qvec=chan_obj.qvec, overflow='wrap', signed=1)
    sample_fi.gen_full_data()
    # gen_output_buffer(chan_obj.Mmax, path)  only used in synthesis bank.
    exp_name, cic_name = gen_exp_shifter(chan_obj, avg_len, path=path)
    inbuff_name, inbuff_cnt_in, inbuff_cnt_out = gen_input_buffer(chan_obj.Mmax, path, )
    circbuff_name = gen_circ_buffer(chan_obj.Mmax, path)
    pfb_name = gen_pfb(chan_obj, path, fs=fs)
    gen_downselect(chan_obj.Mmax, path)
    gen_final_cnt(path)
    print(chan_obj.Mmax)


    return exp_name, cic_name, inbuff_name, circbuff_name, pfb_name

def get_args():

    M_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    avg_len = 256
    chan_file = SIM_PATH + 'chan_out_8_file.bin'
    synth_file = SIM_PATH + 'synth_out_8_file.bin'
    input_file = SIM_PATH + 'sig_store_test8.bin'
    pfb_file = SIM_PATH + 'pfb_out_8_file.bin'
    gen_2X = False
    taps_per_phase = 16
    Mmax = 512
    dsp48e2 = False

    parser = ArgumentParser(description='Channelizer CLI -- Used to generate RTL code, input stimulus, and process output of RTL simulation.')
    parser.add_argument('-l', '--generate_logic', action='store_true', help='Generates RTL Logic for modules, FIFOs, DSP48, CIC filters, etc, to be used in exp_shifter.vhd, input_buffer.vhd, circ_buffer.vhd, and the PFB module')
    parser.add_argument('-c', '--rtl-chan-outfile', type=str, help='Process RTL output file specified by input string -- can use \'default\' as input ')
    parser.add_argument('-s', '--rtl-synth-outfile', type=str, help='Process RTL output file specified by input string -- can use \'default\' as input ')
    parser.add_argument('--check-stim', type=str, help='Plot PSD of input file')
    parser.add_argument('--gen-tones', action='store_true', help='Generate Input Tones')
    parser.add_argument('--process-pfb', type=str, help='Process PFB by running through FFT -- can use \'default\' as input')
    parser.add_argument('--process-inbuff', type=str, help='Plot PSD of input buffer')
    parser.add_argument('--process-input', type=str, help='Generate Channelizer Ouput with Python code')
    parser.add_argument('-i', '--rtl-sim-infile', type=str, help='Generate RTL input and store to filename specified by string -- can use \'default\' as input')
    parser.add_argument('-t', '--generate-taps', action='store_true', help='Generates tap files for all valid FFT Sizes : [8, 16, 32, 64, 128, 256, 512, 1024, 2048]')
    parser.add_argument('-m', '--generate-masks', action='store_true', help='Generate Mask files for all valid FFT Sizes : [8, 16, 32, 64, 128, 256, 512, 1024, 2048]')
    parser.add_argument('-o', '--opt-taps', action='store_true', help='Returns optimized filter parameters all valid FFT Sizes : [8, 16, 32, 64, 128, 256, 512, 1024, 2048]')
    parser.add_argument('--generate-animation', action='store_true', help='Generates polyphase filter animation')
    parser.add_argument('--find-opt-filters', action='store_true', help='Designs optimized Exponential filters')
    parser.add_argument('--gen-2X', action='store_true', help='Flag indicates that a M/2 Channelizer is desired')
    parser.add_argument('--Mmax', type=str, help='Maximum decimation of desired channelizer')
    parser.add_argument('--tps', type=str, help='Specify Taps Per PFB Phase (tps)')
    parser.add_argument('--e2', action='store_true',  help='Specify using DSP48E2 modules')

    args = parser.parse_args()

    if args.e2:
        dsp48e2 = True

    if args.gen_2X:
        gen_2X = True

    if args.Mmax is not None:
        Mmax = int(args.Mmax)

    if args.tps is not None:
        taps_per_phase = int(args.tps)

    if args.gen_tones:
        gen_tones(M=512, lidx=30, ridx=31, offset=0.00050)

    if args.generate_logic:
        qvec_coef = ret_qcoef(dsp48e2)
        chan_obj = Channelizer(M=Mmax, taps_per_phase=taps_per_phase, gen_2X=gen_2X, desired_msb=DESIRED_MSB, qvec=QVEC, 
                               qvec_coef=qvec_coef, fc_scale=FC_SCALE, taps=TAPS)

        sample_fi = fp_utils.ret_dec_fi(0, qvec=QVEC, overflow='wrap', signed=1)
        sample_fi.gen_full_data()
        # gen_output_buffer(Mmax)
        exp_tuple = gen_exp_shifter(chan_obj, avg_len) #, sample_fi=sample_fi)
        # exp_name, cic_name, cic_fifo, slicer_name, exp_fifo
        inbuff_tuple = gen_input_buffer(Mmax, IP_PATH, gen_2X=gen_2X)
        # inbuff_name, inbuff_cnt_in, inbuff_cnt_out, in_fifo, out_fifo = inbuff_tuple
        gen_circ_buffer(Mmax)
        pfb_tuple = gen_pfb(chan_obj)
        down_tuple = gen_downselect(Mmax)
        gen_mux(Mmax)
        final_cnt_name = gen_final_cnt()
        fft_name = f'xfft_{Mmax}'
        chan_name, _ = gen_chan_top(IP_PATH, chan_obj, exp_tuple[0], pfb_tuple[0], fft_name, final_cnt_name)
        if chan_obj.gen_2X:
            copyfile('./verilog/circ_buffer.v', IP_PATH + '/circ_buffer.v')
            copyfile('./verilog/input_buffer.v', IP_PATH + '/input_buffer.v')
        else:
            copyfile('./verilog/input_buffer_1x.v', IP_PATH + '/input_buffer_1x.v')
        print(Mmax)

        gen_do_file(IP_PATH, Mmax, gen_2X, chan_name, exp_tuple, inbuff_tuple, pfb_tuple, down_tuple, final_cnt_name)

    if args.rtl_chan_outfile is not None:
        if args.rtl_chan_outfile.lower() != 'default':
            chan_file = args.rtl_chan_outfile
        process_chan_out(chan_file, Mmax=Mmax, iq_offset=10*taps_per_phase)

    if args.rtl_synth_outfile is not None:
        if args.rtl_synth_outfile.lower() != 'default':
            synth_file = args.rtl_synth_outfile
        process_synth_out(synth_file)

    if args.rtl_sim_infile is not None:
        if args.rtl_sim_infile.lower() != 'default':
            input_file = args.rtl_sim_infile
        gen_test_sig(input_file)

    if args.check_stim is not None:
        if args.check_stim.lower() != 'default':
            input_file = args.check_stim
        plot_input_file(input_file)

    if args.process_pfb is not None:
        if args.process_pfb.lower() != 'default':
            pfb_file = args.process_pfb
        process_pfb_out(pfb_file)

    if args.process_inbuff is not None:
        inbuff_file = args.process_inbuff
        process_inbuff(inbuff_file)

    if args.process_input is not None:
        M = Mmax
        print(f"gen_2x = {gen_2X}")
        print(K_default)
        K_terms = OrderedDict([(8, 7.558734999999983), (16, 7.558734999999983), (128, 7.558734999999983)])
        K_terms = OrderedDict([(8, 7.558734999999983), (16, 15.838734999999983), (128, 15.830899999999993)])
        msb_terms = OrderedDict([(128, 39)])
        offset_terms = OrderedDict([(8, 0.5), (16, 0.5), (128, 0.498)])
        fc_scale = .75

        qvec_coef = ret_qcoef(dsp48e2)

        chan = Channelizer(M=M, gen_2X=gen_2X, qvec=QVEC, qvec_coef=qvec_coef, K_terms=K_terms,
                           offset_terms=offset_terms, desired_msb=39, fc_scale=fc_scale)
        vec = read_complex_samples(args.process_input, q_first=False) * (2 ** -15)
        # vec = np.fromfile(args.process_input, dtype=np.complex64)
        chan_out = chan.analysis_bank(vec, plot_out=False)
        pickle.dump(chan_out, open('float_output.p', 'wb'))
        row_offset = 200
        row_end = 30_000
        resps = []
        wvecs = []
        time_sigs = []
        plot_psd_helper(vec, title='Channelizer Input', savefig=True, dpi=120, plot_time=True)
        title = 'Channelized Output'
        for ii, row in enumerate(chan_out):
            # print(len(row))
            # row = row[row_offset:row_end]
            # print(np.max(np.abs(row)))
            wvec, psd = gen_psd(row, fft_size=256, window='rect')
            resps.append(psd)
            wvecs.append(wvec)
            time_sigs.append(row)
            lg_idx = np.argmax(np.abs(row))
            real_value = np.real(row[lg_idx])
            imag_value = np.imag(row[lg_idx])
            res_value = np.max(psd)
            print("{} : Largest value = {}, i{} - resp = {} db".format(ii, real_value, imag_value, res_value))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(bottom=.10, left=.1, top=.95)
        fig.subplots_adjust(hspace=.50, wspace=.2)
        fig.set_size_inches(12., 12.)
        fig.set_dpi(120)

        fig_time, ax_time = plt.subplots(nrows=1, ncols=2, sharex=True)
        fig_time.subplots_adjust(bottom=.10, left=.1, top=.95)
        fig_time.subplots_adjust(hspace=.50, wspace=.2)
        fig_time.set_size_inches(12., 12.)
        fig_time.set_dpi(120)
        print("wvecs  shape = {}".format(np.shape(wvecs)))
        title = 'Channelizer Output'
        time_title = 'Channelizer Output time'
        chan_list = [4, 5, 48, 96, 97, 98, 123, 124]
        marker_idx = 0
        style_list = ['solid', 'dash', 'dashdot']
        style_idx = 0
        msize = 4
        for idx in range(M):
            if idx not in chan_list:
                continue
            label = f"bin {idx}"
            # print(label)
            marker = marker_list[marker_idx]
            style = style_list[style_idx]
            plot_psd(ax, wvecs[idx], resps[idx], label=[label], miny=-100, maxy=10, legendsize=8, legend_cols=2, marker=marker, markersize=msize, linestyle=[style], title=r'$\sf{Chan\ Output}$')
            plot_time_sig(ax_time[0], np.real(time_sigs[idx]), label=[label], miny=-1., maxy=1., legendsize=8, legend_cols=2, marker=marker, markersize=msize, linestyle=[style])
            plot_time_sig(ax_time[1], np.imag(time_sigs[idx]), label=[label], title=r'$\sf{Imag}$', miny=-1.,maxy=1., legendsize=8, legend_cols=2, marker=marker, markersize=msize, linestyle=[style])
            if marker_idx + 1 == len(marker_list):
                style_idx = (style_idx + 1) % len(style_list)
            marker_idx = (marker_idx + 1) % len(marker_list)
            
    # plot_psd(ax[0][1], wvecs[42], resps[42], title='Channel 1', miny=-120, maxy=10)
    # plot_psd(ax[1][0], wvecs[43], resps[41], title='Channel 2', miny=-120, maxy=10)
    # plot_psd(ax[1][1], wvecs[44], resps[44], title='Channel 3', miny=-120, maxy=10)
    # plot_psd(ax[2][0], wvecs[45], resps[45], title='Channel 4', miny=-120, maxy=10)
    # plot_psd(ax[2][1], wvecs[46], resps[46], title='Channel 5', miny=-120, maxy=10)

        file_name = copy.copy(title)
        file_name = ''.join(e if e.isalnum() else '_' for e in file_name)
        file_name += '.png'
        file_name = file_name.replace("__", "_")
        print(file_name)
        fig.savefig(file_name)

        file_name = copy.copy(time_title)
        file_name = ''.join(e if e.isalnum() else '_' for e in file_name)
        file_name += '.png'
        file_name = file_name.replace("__", "_")
        print(file_name)
        fig_time.savefig(file_name)
    if args.generate_taps:
        gen_tap_plots(M_list)

    if args.generate_masks:
        gen_mask_files(M_list)

    if args.opt_taps:
        populate_fil_table()

    if args.generate_animation:
        chan_obj = Channelizer(M=4, gen_2X=False)
        chan_obj.gen_animation()

    if args.find_opt_filters:
        find_best_terms()

    return 


if __name__ == "__main__":

    # chan_list = [4, 48, 96, 97, 98, 123, 124]
    # sig_bws = np.linspace(.001, .004, len(chan_list)).tolist()
    # amps = np.linspace(.2, .5, len(chan_list)).tolist()
    # roll_values = np.linspace(1000, 5000, len(chan_list))
    # roll_values = [int(value) for value in roll_values]
    # gen_test_sig(128, './rw_test.bin', False, chan_list, sig_bws, amps, roll_values)

    modem_args = get_args()
    plt.show(block=blockl)
