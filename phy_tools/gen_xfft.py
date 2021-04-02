#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from phy_tools.gen_utils import ret_valid_path
from phy_tools.channelizer import calc_fft_tuser_width

def gen_xfft_xci(path=None, max_fft_size=2048, arch='virtex7', device='xc7vx485t', pkg='ffg1157', speed_grade='-1',
                 phase_factor_width=25, bram_stages=4, sw_version='2018.2'):

    assert(path is not None), 'User must specify Path'
    path = ret_valid_path(path)

    nfft_max = int(np.floor(np.log2(max_fft_size)))
    max_fft_size = 1 << nfft_max

    if max_fft_size < 8:
        max_fft_size = 8
        nfft_max = 3
    
    if max_fft_size > (1 << 16):
        max_fft_size = 1 << 16
        nfft_max = 16

    mod_name = 'xfft_{}'.format(max_fft_size)
    
    if path is not None:
        file_name = path + mod_name + '.xci'
    else:
        file_name = './' + mod_name + '.xci'

    tuser_bits = calc_fft_tuser_width(max_fft_size)


    # calculate max bram stages. if user exceeds then make it that.
    max_bram_stages = np.max((nfft_max - 5, 0))
    bram_stages = max_bram_stages if bram_stages > max_bram_stages else bram_stages

    with open(file_name, 'w') as fh:

        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write('<spirit:design xmlns:xilinx="http://www.xilinx.com" xmlns:spirit="http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
        fh.write('  <spirit:vendor>xilinx.com</spirit:vendor>\n')
        fh.write('  <spirit:library>xci</spirit:library>\n')
        fh.write('  <spirit:name>unknown</spirit:name>\n')
        fh.write('  <spirit:version>1.0</spirit:version>\n')
        fh.write('  <spirit:componentInstances>\n')
        fh.write('    <spirit:componentInstance>\n')
        fh.write('      <spirit:instanceName>{}</spirit:instanceName>\n'.format(mod_name))
        fh.write('      <spirit:componentRef spirit:vendor="xilinx.com" spirit:library="ip" spirit:name="xfft" spirit:version="9.1"/>\n')
        fh.write('      <spirit:configurableElementValues>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.ACLKEN_INTF.POLARITY">ACTIVE_LOW</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.ACLK_INTF.CLK_DOMAIN"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.ACLK_INTF.FREQ_HZ">100000000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.ACLK_INTF.PHASE">0.000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_DATA_IN_CHANNEL_HALT_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_DATA_OUT_CHANNEL_HALT_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_FFT_OVERFLOW_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_FRAME_STARTED_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_STATUS_CHANNEL_HALT_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_TLAST_MISSING_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.EVENT_TLAST_UNEXPECTED_INTF.PortWidth">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.CLK_DOMAIN"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.FREQ_HZ">100000000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TKEEP">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TLAST">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TREADY">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TSTRB">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.LAYERED_METADATA">undef</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.PHASE">0.000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TDATA_NUM_BYTES">4</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TDEST_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TID_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TUSER_WIDTH">{}</spirit:configurableElementValue>\n'.format(tuser_bits))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.CLK_DOMAIN"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.FREQ_HZ">100000000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TKEEP">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TLAST">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TREADY">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TSTRB">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.LAYERED_METADATA">undef</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.PHASE">0.000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TDATA_NUM_BYTES">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TDEST_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TID_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TUSER_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.CLK_DOMAIN"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.FREQ_HZ">100000000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TKEEP">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TLAST">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TREADY">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TSTRB">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.LAYERED_METADATA">undef</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.PHASE">0.000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TDATA_NUM_BYTES">2</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TDEST_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TID_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TUSER_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.CLK_DOMAIN"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.FREQ_HZ">100000000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TKEEP">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TLAST">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TREADY">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TSTRB">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.LAYERED_METADATA">undef</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.PHASE">0.000</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TDATA_NUM_BYTES">4</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TDEST_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TID_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TUSER_WIDTH">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_ARCH">3</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_BFLY_TYPE">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_BRAM_STAGES">{}</spirit:configurableElementValue>\n'.format(bram_stages))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_CHANNELS">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_CMPY_TYPE">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_DATA_MEM_TYPE">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_ACLKEN">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_ARESETN">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_BFP">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_CYCLIC_PREFIX">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_NATURAL_INPUT">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_NATURAL_OUTPUT">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_NFFT">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_OVFLO">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_ROUNDING">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_SCALING">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_HAS_XK_INDEX">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_INPUT_WIDTH">16</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_M_AXIS_DATA_TDATA_WIDTH">32</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_M_AXIS_DATA_TUSER_WIDTH">{}</spirit:configurableElementValue>\n'.format(tuser_bits))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_M_AXIS_STATUS_TDATA_WIDTH">8</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_NFFT_MAX">{}</spirit:configurableElementValue>\n'.format(nfft_max))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_OPTIMIZE_GOAL">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_OUTPUT_WIDTH">16</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_REORDER_MEM_TYPE">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_S_AXIS_CONFIG_TDATA_WIDTH">16</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_S_AXIS_DATA_TDATA_WIDTH">32</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_THROTTLE_SCHEME">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_TWIDDLE_MEM_TYPE">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_TWIDDLE_WIDTH">20</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_USE_FLT_PT">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_USE_HYBRID_RAM">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="MODELPARAM_VALUE.C_XDEVICEFAMILY">kintex7</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.Component_Name">{}</spirit:configurableElementValue>\n'.format(mod_name))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.aclken">false</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.aresetn">true</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.butterfly_type">use_luts</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.channels">1</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.complex_mult_type">use_mults_resources</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.cyclic_prefix_insertion">false</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.data_format">fixed_point</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.implementation_options">pipelined_streaming_io</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.input_width">16</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.memory_options_data">block_ram</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.memory_options_hybrid">false</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.memory_options_phase_factors">block_ram</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.memory_options_reorder">block_ram</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.number_of_stages_using_block_ram_for_data_and_phase_factors">{}</spirit:configurableElementValue>\n'.format(bram_stages))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.output_ordering">natural_order</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.ovflo">false</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.phase_factor_width">{}</spirit:configurableElementValue>\n'.format(phase_factor_width))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.rounding_modes">convergent_rounding</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.run_time_configurable_transform_length">true</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.scaling_options">block_floating_point</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.target_clock_frequency">250</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.target_data_throughput">50</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.throttle_scheme">nonrealtime</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.transform_length">{}</spirit:configurableElementValue>\n'.format(max_fft_size))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PARAM_VALUE.xk_index">true</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.ARCHITECTURE">{}</spirit:configurableElementValue>\n'.format(arch))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.BOARD"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.DEVICE">{}</spirit:configurableElementValue>\n'.format(device))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.PACKAGE">{}</spirit:configurableElementValue>\n'.format(pkg))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.PREFHDL">VERILOG</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.SILICON_REVISION"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.SIMULATOR_LANGUAGE">MIXED</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.SPEEDGRADE">{}</spirit:configurableElementValue>\n'.format(speed_grade))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.TEMPERATURE_GRADE"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.USE_RDI_CUSTOMIZATION">TRUE</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="PROJECT_PARAM.USE_RDI_GENERATION">TRUE</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.IPCONTEXT">IP_Flow</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.IPREVISION">0</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.MANAGED">TRUE</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.OUTPUTDIR">.</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.SELECTEDSIMMODEL"/>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.SHAREDDIR">.</spirit:configurableElementValue>\n')
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.SWVERSION">{}</spirit:configurableElementValue>\n'.format(sw_version))
        fh.write('        <spirit:configurableElementValue spirit:referenceId="RUNTIME_PARAM.SYNTHESISFLOW">OUT_OF_CONTEXT</spirit:configurableElementValue>\n')
        fh.write('      </spirit:configurableElementValues>\n')
        fh.write('      <spirit:vendorExtensions>\n')
        fh.write('        <xilinx:componentInstanceExtensions>\n')
        fh.write('          <xilinx:configElementInfos>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TKEEP" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TLAST" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.HAS_TSTRB" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TDATA_NUM_BYTES" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TDEST_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TID_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_DATA.TUSER_WIDTH" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TKEEP" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TLAST" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.HAS_TSTRB" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TDEST_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TID_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.M_AXIS_STATUS.TUSER_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TKEEP" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TLAST" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TREADY" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.HAS_TSTRB" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TDATA_NUM_BYTES" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TDEST_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TID_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_CONFIG.TUSER_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TKEEP" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TLAST" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TREADY" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.HAS_TSTRB" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TDATA_NUM_BYTES" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TDEST_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TID_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="BUSIFPARAM_VALUE.S_AXIS_DATA.TUSER_WIDTH" xilinx:valueSource="constant"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.aresetn" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.implementation_options" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.number_of_stages_using_block_ram_for_data_and_phase_factors" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.output_ordering" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.phase_factor_width" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.rounding_modes" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.run_time_configurable_transform_length" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.scaling_options" xilinx:valueSource="user"/>\n')
        # fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.throttle_scheme" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.transform_length" xilinx:valueSource="user"/>\n')
        fh.write('            <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.xk_index" xilinx:valueSource="user"/>\n')
        fh.write('          </xilinx:configElementInfos>\n')
        fh.write('        </xilinx:componentInstanceExtensions>\n')
        fh.write('      </spirit:vendorExtensions>\n')
        fh.write('    </spirit:componentInstance>\n')
        fh.write('  </spirit:componentInstances>\n')
        fh.write('</spirit:design>\n')

    return mod_name