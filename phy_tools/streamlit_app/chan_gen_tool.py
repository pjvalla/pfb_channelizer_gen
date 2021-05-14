import streamlit as st
import pandas as pd
import numpy as np
from phy_tools.channelizer import Channelizer, gen_exp_shifter, gen_input_buffer, gen_circ_buffer, gen_pfb, gen_downselect, ret_qcoef
from phy_tools.channelizer import gen_final_cnt, gen_output_buffer, populate_fil_table, gen_mux, gen_mask_files, gen_tones_vec
from phy_tools.channelizer import ret_qcoef
from phy_tools.chan_utils import gen_chan_top, gen_chan_tb, gen_do_file

from gen_buffers import gen_inbuff_logic, gen_inbuff1x_logic, gen_circbuff_logic, gen_reader, gen_writer
import phy_tools.adv_pfb as adv_filter
from phy_tools.plotly_utils import plotly_time_helper
from phy_tools.gen_xfft import gen_xfft_xci
from phy_tools.plt_utils import find_pwr
from phy_tools.fp_utils import dec_to_hex
import SessionState

import pandas as pd
from collections import OrderedDict
import ipdb

from shutil import  make_archive
import base64

import os

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, _, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def get_download_link(file_name, path=None, file_type='zip'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    # b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/zip;base64,{b64}">Download zip file</a>'

    file_name = path + file_name if path is not None else file_name
    with open(file_name, 'rb') as fin:
        bytes_var = fin.read()
        b64 = base64.b64encode(bytes_var).decode()
    return f'<a href="data:file/zip;base64,{b64}" download="{file_name}">Download {file_type} file</a>'

QVEC = (16, 15)
IP_PATH = './tmp/'
AVG_LEN = 256

st.title('Channelizer Verilog Generation Tool')

fft_list = tuple(2 ** np.arange(3, 17))
db_list = tuple(np.arange(-60, -300, -20))
gen_2X = False

# fc_scale = st.
max_fft = st.sidebar.selectbox("Maximum Number of Channels", fft_list, index=3)
taps_per_phase = st.sidebar.number_input("Taps Per Arm of PFB", value=16, min_value=12, max_value=64)
chan_type = st.sidebar.radio("Channelizer Type", ('M', 'M/2'))
dsp_type = st.sidebar.radio("DSP Type", ('DSP48E1', 'DSP48E2'))
min_db = st.sidebar.selectbox("Plot Min DB", db_list, index=3)
fc_scale = st.sidebar.number_input("Cut off Frequency (Proportional to Bin Width)", value=.75, min_value=0.5, max_value=1.0)
tbw_scale = st.sidebar.number_input("Transition Bandwidth (Proportional to Bin Width) - M/2 Designs should relax specs", value=.20, min_value=0.09, max_value=1.0)

K_orig = OrderedDict([(8, 9.169244999999984), (16, 9.169244999999984), (32, 9.169244999999984),
                      (64, 9.169244999999984), (128, 9.169244999999984), (256, 9.169244999999984), (512, 9.169244999999984),
                      (1024, 9.169244999999984), (2048, 9.169244999999984), (4096, 9.169244999999984),
                      (8192, 9.169244999999984), (16384, 9.169244999999984), (32768, 9.169244999999984), (65536, 9.169244999999984)])

FFT_SIZES = [value for value in K_orig.keys()]

msb_orig = OrderedDict([(8, 39), (16, 39), (32, 39), (64, 39), (128, 39), (256, 39), (512, 39), (1024, 39), 
                         (2048, 39), (4096, 39), (8192, 39), (16384, 39), (32768, 39), (65536, 39)])
offset_orig = OrderedDict([(8, .5), (16, .5), (32, .5), (64, .5), (128, .5), (256, .5), (512, 0.5), (1024, 0.5), 
                            (2048, 0.5), (4096, 0.5), (8192, 0.5), (16384, 0.5), (32768, 0.5), (65536, 0.5)])

session_state = SessionState.get(K_terms=K_orig, msb_terms=msb_orig, offset_terms=offset_orig)

dsp48e2 = True if dsp_type == 'DSP48E2' else False
qvec_coef = ret_qcoef(dsp48e2)

def update_chan_obj(session_state): 

    try:
        return Channelizer(M=max_fft, taps_per_phase=taps_per_phase, gen_2X=gen_2X, qvec=QVEC,
                           qvec_coef=qvec_coef, fc_scale=fc_scale, tbw_scale=tbw_scale, 
                           K_terms=session_state.K_terms, offset_terms=session_state.offset_terms, 
                           desired_msb=session_state.msb_terms[max_fft])
    except:
        session_state.K_terms = K_orig
        session_state.msb_terms = msb_orig
        session_state.offset_terms = offset_orig
    finally:
        return Channelizer(M=max_fft, taps_per_phase=taps_per_phase, gen_2X=gen_2X, qvec=QVEC,
                           qvec_coef=qvec_coef, fc_scale=fc_scale, tbw_scale=tbw_scale,
                           K_terms=session_state.K_terms, offset_terms=session_state.offset_terms,
                           desired_msb=session_state.msb_terms[max_fft])


    # remove all .v and .xci files from IP_PATH
def remove_files(path):
    ip_files = os.listdir(path)
    for item in ip_files:
        if item.endswith(".v") or item.endswith(".xci") or item.endswith(".sv") or item.endswith(".bin") or item.endswith(".do"):
            os.remove(os.path.join(path, item))

if chan_type == 'M/2':
    gen_2X = True
    chan_str = 'M_2'
else:
    gen_2X = False
    chan_str = 'M'

st.sidebar.markdown("""
<style>
.small-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="big-font">Click Optimize Filter after updating filter settings</p>', unsafe_allow_html=True)
opt_button = st.sidebar.button('Optimize Filter')
gen_button = st.sidebar.button('Generate Verilog')
gen_taps_button = st.sidebar.button('Generate Taps -- Pandas pickle')

def opt_params(): 
    # st.write("Optimizing Taps = {}".format(opt_button))
    m_fft = 64  #max_fft if max_fft < 128 else 64
    return populate_fil_table(start_size=m_fft, end_size=m_fft, fc_scale=fc_scale, qvec=QVEC, qvec_coef=qvec_coef,
                              taps_per_phase=taps_per_phase, gen_2X=gen_2X, tbw_scale=tbw_scale, freqz_pts=1000)

if opt_button:
    K_term, msb_term, offset_term = opt_params() 
    kterm = next(iter(K_term.values()))
    msbterm = next(iter(msb_term.values()))
    offterm = next(iter(offset_term.values()))
    # copy terms of other sizes.
    K_terms = []
    msb_terms = []
    offset_terms = []
    for fft_size in FFT_SIZES:
        K_terms.append((fft_size, kterm))
        msb_terms.append((fft_size, msbterm))
        offset_terms.append((fft_size, offterm))
    session_state.K_terms = OrderedDict(K_terms)
    session_state.msb_terms = OrderedDict(msb_terms)
    session_state.offset_terms = OrderedDict(offset_terms)


# @st.cache
def update_psd(session_state): 
    # print("updating PSD")
    chan_obj = update_chan_obj(session_state) 
    fft_size = 2048
    minx = -4. / max_fft
    maxx = 4. / max_fft
    step = (maxx - minx) / fft_size
    freq_vector = np.arange(minx, maxx, step)
    # this vector is normalized frequency.
    resp, omega = chan_obj.gen_psd(freq_vector=freq_vector, fixed_taps=True)
    data_dict = {'Omega': omega, 'PSD':resp, 'sig_idx': 0}
    return pd.DataFrame(data_dict)


def gen_taps(session_state): 
    ret_df = pd.DataFrame()
    max_taps = max_fft * taps_per_phase
    for fft_size in fft_list:
    # generate all taps upto max_fft.
        chan_obj = update_chan_obj(session_state) 
        pad = max_taps - len(chan_obj.taps)
        taps = np.pad(chan_obj.taps, (0, pad), mode='constant', constant_values=0.)
        fixed_taps = np.pad(chan_obj.taps_fi.flatten(), (0, pad), mode='constant', constant_values=0)
        hex_taps = dec_to_hex(fixed_taps, 8)  # 32 bit values
        poly_taps = np.reshape(taps, (max_fft, -1), order='F').T.flatten()
        fixed_poly_taps = np.reshape(fixed_taps, (max_fft, -1), order='F').T.flatten()
        hex_poly_taps = np.reshape(hex_taps, (max_fft, -1), order='F').T.flatten()
        data = {'FFT Size':fft_size, 'Float Taps':taps, 'Fixed Taps':fixed_taps, 'Hex Taps':hex_taps,
                'Float Taps Reshaped':poly_taps, 'Fixed Taps Reshaped': fixed_poly_taps, 'Hex Taps Reshaped':hex_poly_taps}

        temp_df = pd.DataFrame(data)
        temp_df = temp_df.set_index('FFT Size')
        ret_df = ret_df.append(temp_df)
        if fft_size == max_fft:
            break
    file_name = 'taps'
    full_name = file_name + '.p'
    ret_df.to_pickle(full_name)
    st.sidebar.markdown(get_download_link(full_name, path=None, file_type='pickle'), unsafe_allow_html=True)


# @st.cache
def update_stem(session_state): 
    chan_obj = update_chan_obj(session_state)
    taps = chan_obj.taps
    data_dict = {'Taps': taps, 'sig_idx': 0}
    return pd.DataFrame(data_dict)

if gen_taps_button:
    gen_taps(session_state)


if gen_button:
    # check if IP_PATH exists -- create it if it doesn't
    if not os.path.exists(IP_PATH):
        os.makedirs(IP_PATH)
    remove_files(IP_PATH)           
    chan_obj = update_chan_obj(session_state)
    print("===========================================================")
    print("Generate K {}, Mmax {}".format(chan_obj.K, chan_obj.Mmax))
    print("===========================================================")
    gen_output_buffer(max_fft, IP_PATH)
    exp_tuple = gen_exp_shifter(chan_obj, AVG_LEN, path=IP_PATH)
    inbuff_tuple = gen_input_buffer(max_fft, IP_PATH, gen_2X)
    gen_circ_buffer(max_fft, IP_PATH)
    pfb_tuple = gen_pfb(chan_obj, path=IP_PATH, dsp48e2=dsp48e2)
    gen_mux(max_fft, IP_PATH)
    down_tuple = gen_downselect(max_fft, IP_PATH)
    final_cnt_name = gen_final_cnt(IP_PATH)
    fft_name = gen_xfft_xci(IP_PATH, max_fft)
    tones = gen_mask_files([max_fft], percent_active=.25, path=IP_PATH)
    # generate every other tone.
    tone_vec = tones[0][::2]
    tstep = .65 / (len(tone_vec) * max_fft)
    offset_vec = [i * tstep  for i in range(len(tone_vec))]
    gen_tones_vec(tone_vec, M=max_fft, offset=offset_vec, path=IP_PATH)

    chan_name, _ = gen_chan_top(IP_PATH, chan_obj, exp_tuple[0], pfb_tuple[0], fft_name, final_cnt_name)
    gen_chan_tb(IP_PATH, chan_obj, len(tones[0]))
    adv_filter.gen_sim_vh(IP_PATH)

    gen_do_file(IP_PATH, max_fft, gen_2X, chan_name, exp_tuple, inbuff_tuple, pfb_tuple, down_tuple, final_cnt_name)
    if chan_obj.gen_2X:
        gen_inbuff_logic(IP_PATH)
        gen_circbuff_logic(IP_PATH)
    else:
        gen_inbuff1x_logic(IP_PATH)

    gen_reader(IP_PATH)
    gen_writer(IP_PATH)

    file_name = 'verilog_{}_{}_{}'.format(max_fft, taps_per_phase, chan_str)
    # full_name = IP_PATH + '/' + file_name
    path = IP_PATH
    make_archive(file_name, 'zip', IP_PATH)
    print(file_name)
    st.sidebar.markdown(get_download_link(file_name + '.zip', file_type='zip'), unsafe_allow_html=True)
else:
    pass
    # st.write('New Channelizer')


psd_df = update_psd(session_state) #, taps_per_phase, gen_2X, max_fft, dsp_type)
# try:

minx = -4. / max_fft
maxx = 4. / max_fft
pwr_pt = -3.01 * 2

binl = -1. / max_fft
binr = 1. / max_fft
bin_width = binr - binl
# xlabel = st.latex(r'''\pi\ rads/sec''')
# 'Discrete Freq.'
fig = plotly_time_helper(psd_df, opacity=[.8] * 2, miny=min_db, maxy=10, index_str='sig_idx', x_name='Omega', y_name='PSD',
                            labelsize=20, titlesize=30, xlabel='\u03A0 rads/sec', ylabel='dB', 
                            subplot_title=('PSD',), minx=minx, maxx=maxx) #, pwr_pts=-3.01)

resp = psd_df['PSD'].to_numpy()
lidx, ridx = find_pwr(resp, pwr_pt)
pass_right = psd_df['Omega'].iloc[ridx]
pass_left = psd_df['Omega'].iloc[lidx]
# pass_bw = int(bin_width * len(resp))
tbw_right = pass_right + bin_width * tbw_scale
tbw_left = pass_left - bin_width * tbw_scale
bound_left = binl - .5 * bin_width
bound_right = binr + .5 * bin_width

tbw_left_idx = np.where(psd_df['Omega'] < tbw_left)[0][-1]
stop_atten = np.max(resp[tbw_left_idx-30: tbw_left_idx])
x_left = psd_df['Omega'].iloc[lidx]
x_right = psd_df['Omega'].iloc[ridx]
fig.add_shape(type="line", x0=x_left, y0=-1000, x1=x_left, y1=1000,
                line=dict(color="forestgreen", width=2, dash='dash'))
fig.add_shape(type="line", x0=x_right, y0=-1000, x1=x_right, y1=1000,
                line=dict(color="forestgreen", width=2, dash='dash'))
# tbw_left = psd_df['Omega'].iloc[resp_offset_lidx]
# tbw_right = psd_df['Omega'].iloc[resp_offset_ridx]
trans_bw = tbw_right - x_right 
fig.add_shape(type="line", x0=tbw_left, y0=-1000, x1=tbw_left, y1=1000,
                line=dict(color="magenta", width=2, dash='dash'))
fig.add_shape(type="line", x0=tbw_right, y0=-1000, x1=tbw_right,
                y1=1000, line=dict(color="magenta", width=2, dash='dash'))
fig.add_shape(type="line", x0=binl, y0=-1000, x1=binl,
                y1=1000, line=dict(color="darkgreen", width=2, dash='dash'))
fig.add_shape(type="line", x0=binr, y0=-1000, x1=binr,
                y1=1000, line=dict(color="darkgreen", width=2, dash='dash'))
fig.add_shape(type="line", x0=bound_left, y0=-1000, x1=bound_left,
                y1=1000, line=dict(color="crimson", width=2, dash='dash'))
fig.add_shape(type="line", x0=bound_right, y0=-1000, x1=bound_right,
                y1=1000, line=dict(color="crimson", width=2, dash='dash'))
fig.add_shape(type="line", x0=-100, y0=stop_atten, x1=100, y1=stop_atten, line=dict(color="crimson",width=2, dash='dash'))

offset = int((10 - min_db) * .03)
fig.add_annotation(
    x=0,
    y=stop_atten + 40 + offset,
    xref="x",
    yref="y",
    text="1 Bin Width",
    font=dict(
        family='sans serif',
        size=14,
        color="darkgreen"
    ),
    align='center',
    showarrow=False,
)
fig.add_annotation(
    x=binl,
    y=stop_atten + 40,
    xref="x",
    yref="y",
    text="",
    ax = 0,
    ay = stop_atten + 40,
    axref = "x", 
    ayref = "y",
    arrowhead = 3,
    arrowwidth = 1.5,
    arrowcolor="darkgreen",
    opacity=1.0,
)
fig.add_annotation(
    x=binr,
    y=stop_atten + 40,
    xref="x",
    yref="y",
    ax = .03 * binl,
    ay = stop_atten + 40,
    text = "",
    axref = "x", 
    ayref = "y",
    arrowhead = 3,
    arrowwidth = 1.5,
    arrowcolor="darkgreen",
    opacity=1.0,
)

fig.add_annotation(
    x=0,
    y=stop_atten + 30,
    xref="x",
    yref="y",
    text="Adj. Bin Boundary",
    font=dict(
        family='sans serif',
        size=15,
        color="crimson"
    ),
    align='center',
    showarrow=False,
)
fig.add_annotation(
    x=bound_left,
    y=stop_atten + 30,
    xref="x",
    yref="y",
    text="",
    ax=bound_left * .60,
    ay=stop_atten + 30,
    axref="x", 
    ayref="y",
    arrowhead = 3,
    arrowwidth = 2.0,
    arrowcolor="crimson",
    opacity=1.,
)
fig.add_annotation(
    x=bound_right,
    y=stop_atten + 30,
    xref="x",
    yref="y",
    text="",
    ax = bound_right * .60, 
    ay = stop_atten + 30,
    axref = "x", 
    ayref = "y",
    arrowhead = 3,
    arrowwidth = 2.0,
    arrowcolor="crimson",
    opacity=1.,
)

fig.add_annotation(
    x=.02, #x_left,
    y=.98,
    xref="paper",
    yref="paper",
    # text="3 dB Freqs = {:.5f}, {:.5f} \u03A0 rads/sec <br> Trans BW = {:.5f} \u03A0 rads/sec <br> Stopband Atten = {:.0f} dB <br> Bin Width = {:.5f} \u03A0 rads/sec <br> 2X Bin Width = {:.5f} \u03A0 rads/sec".format(
    #     x_left, x_right, trans_bw, stop_atten, bin_width, 2*bin_width),
    text="3 dB Freqs = {:.5f}, {:.5f} \u03A0 rads/sec <br> Trans BW = {:.5f} \u03A0 rads/sec <br> Stopband Atten = {:.0f} dB".format(
        x_left, x_right, trans_bw, stop_atten),
    showarrow=False,
    font=dict(
        family='sans serif', 
        size=13,
        color="#ffffff"
    ),
    align="center",
    # arrowhead=2,
    # arrowsize=1,
    # arrowwidth=2,
    # arrowcolor="#636363",
    # ax=20,
    # ay=-30,
    bordercolor="#c7c7c7",
    borderwidth=2,
    borderpad=4,
    bgcolor="#ff7f0e",
    opacity=0.7
)
fig.update_shapes(dict(xref='x', yref='y'))

config = dict({
    "modeBarButtonsToRemove": ['autoScale2d', 'toggleSpikelines'],
})
fig.update_layout(autosize=False, width=900, height=550, margin=dict(l=20, r=40, b=40, t=70))
st.plotly_chart(fig, config= config)

stem_df = update_stem(session_state) #, taps_per_phase, gen_2X, max_fft, dsp_type)
try:
    # ipdb.set_trace()
    num_rows = len(stem_df)
    if num_rows > 1024:
        div = num_rows // 1024
        stem_df = stem_df.iloc[::div, :]

    fig = plotly_time_helper(stem_df, opacity=[.8] * 2, index_str='sig_idx', y_name='Taps', stem_plot=False, miny=-.5,
                             labelsize=20, titlesize=30, xlabel='Tap Index', ylabel='Amplitude', subplot_title=('Taps',))

    fig.update_layout(autosize=False, width=900, height=350, margin=dict(l=20, r=40, b=40, t=70))
    config = dict({
        "modeBarButtonsToRemove": ['toggleSpikelines'],
    })
    st.plotly_chart(fig, config=config)
except:
    pass


