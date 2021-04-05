#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: pjvalla
"""
import os

import numpy as np

from itertools import count
import copy
import ipdb  # analysis:ignore

from phy_tools import fp_utils
from phy_tools.gen_utils import ret_module_name, ret_file_name, ret_valid_path, make_log_tables
from phy_tools.gen_utils import print_header_vhd as print_header
from phy_tools.gen_utils import print_libraries, print_intro, print_exit, print_entity_intro
from phy_tools.vhdl_gen_xilinx import gen_dsp48E1
import phy_tools.vhdl_gen_xilinx as x_utils
from phy_tools.demod_utils import AGCModule

import datetime

from subprocess import check_output, CalledProcessError, DEVNULL
try:
    __version__ = check_output('git log -1 --pretty=format:%cd --date=format:%Y.%m.%d'.split(), stderr=DEVNULL).decode()
except CalledProcessError:
    from datetime import date
    today = date.today()
    __version__ = today.strftime("%Y.%m.%d")


def logic_rst(fh, prefix='a_d', cnt=1, sp=''):
    for jj in range(cnt):
        fh.write('{}{}({}) <= (others => \'0\');\n'.format(sp, prefix, jj))

def logic_gate(fh, prefix='a_d', str_val='a', cnt=1, sp=''):
    for jj in range(cnt):
        rside = str_val if (jj == 0) else '{}[{}]'.format(prefix, jj - 1)
        fh.write('{}{}[{}] <= {};\n'.format(sp, prefix, jj, rside))

def print_sens_list(fh, sig_list):
    last_idx = len(sig_list) - 1
    for idx, signal in enumerate(sig_list):
        if idx == 0:
            fh.write('process({}'.format(signal))
        else:
            fh.write(', {}'.format(signal))

        if idx == last_idx:
            fh.write(')\n')

def ret_mult_eight(input_val):
    return int(np.ceil(input_val / 8.)) * 8

def gen_cnt_rst(fh, prefix='cnt', pdelay=2, sp='', reset_list=None):
    for jj in range(pdelay):
        if reset_list is None:
            fh.write('{}{}_nib{} <= (others => \'0\');\n'.format(sp, prefix, jj))
        else:
            fh.write('{}{}_nib{} <= {};\n'.format(sp, prefix, jj, reset_list[jj]))

def gen_cnt_sigs(fh, prefix='cnt', pdelay=2):
    for jj in range(pdelay):
        fh.write('signal {}_nib{}, next_{}_nib{} : unsigned(8 downto 0);\n'.format(prefix, jj, prefix, jj))

    fh.write('\n')
    for jj in range(pdelay - 1):
        for nn in range(pdelay - jj - 1):
            fh.write('signal {}_nib{}_d{} : unsigned(7 downto 0);\n'.format(prefix, jj, nn))
        fh.write('\n')

def gen_cnt_regs(fh, prefix='cnt', pdelay=2):
    for jj in range(pdelay):
        fh.write('            {}_nib{} <= next_{}_nib{};\n'.format(prefix, jj, prefix, jj))


def gen_cnt_delay(fh, prefix='cnt', pdelay=2, tab=''):
    for jj in range(pdelay - 1):
        for nn in range(pdelay - jj - 1):
            if nn == 0:
                fh.write('        {}{}_nib{}_d{} <= {}_nib{}(7 downto 0);\n'.format(tab, prefix, jj, nn, prefix, jj))
            else:
                fh.write('        {}{}_nib{}_d{} <= {}_nib{}_d{};\n'.format(tab, prefix, jj, nn, prefix, jj, nn - 1))
        fh.write('\n')

def gen_cnt_fback(fh, prefix='cnt', pdelay=2):
    for jj in range(pdelay):
            fh.write('    next_{}_nib{} <= {}_nib{};\n'.format(prefix, jj, prefix, jj))

def adder_pipeline(cnt_width):
    return int(np.ceil((cnt_width - 1) / 8.))

def ret_addr_width(depth):
    fifo_depth = 2 ** int(np.ceil(np.log2(depth)))
    if fifo_depth < 8:
        fifo_depth = 8
    fifo_addr_width = int(np.log2(fifo_depth))

    return fifo_addr_width

def name_help(module_name, path=None):
    if path is not None:
        file_name = path + '/' + module_name + '.vhd'
    else:
        file_name = './' + module_name + '.vhd'

    return file_name

def axi_fifo_inst(fh, fifo_name, data_width, addr_width, af_thresh=None, ae_thresh=None, inst_name='u_fifo', tuser_width=0, tlast=False,
                  s_tvalid_str='s_axis_tvalid', s_tdata_str='s_axis_tdata', s_tlast_str='s_axis_tlast', max_delay=0,
                  s_tuser_str='s_axis_tuser', s_tready_str='open', almost_full_str='almost_full', delay_str=None,
                  m_tvalid_str='m_axis_tvalid', m_tlast_str='m_axis_tlast', m_tuser_str='m_axis_tuser',
                  m_tdata_str='m_axis_tdata', m_tready_str='m_axis_tready'):

    fh.write('{} : {}\n'.format(inst_name, fifo_name))
    fh.write('generic map\n')
    fh.write('(\n')
    fh.write('    DATA_WIDTH => {},\n'.format(data_width))
    if af_thresh is not None:
        fh.write('    ALMOST_FULL_THRESH => {},\n'.format(af_thresh))
    if ae_thresh is not None:
        fh.write('    ALMOST_EMTPY_THRESH => {},\n'.format(ae_thresh))
    if tuser_width > 0:
        fh.write('    TUSER_WIDTH => TUSER_WIDTH,\n')

    fh.write('    ADDR_WIDTH => {}\n'.format(addr_width))
    fh.write(')\n')
    fh.write('port map\n')
    fh.write('(\n')
    fh.write('    clk => clk, \n')
    fh.write('    sync_reset => sync_reset,\n')
    fh.write('\n')
    fh.write('    s_axis_tvalid => {},\n'.format(s_tvalid_str))
    fh.write('    s_axis_tdata => {},\n'.format(s_tdata_str))
    if tlast:
        fh.write('    s_axis_tlast => {},\n'.format(s_tlast_str))
    if tuser_width:
        fh.write('    s_axis_tuser => {},\n'.format(s_tuser_str))
    fh.write('    s_axis_tready => {},\n'.format(s_tready_str))
    if af_thresh is not None:
        fh.write('\n')
        fh.write('    almost_full => {},\n'.format(almost_full_str))
    if max_delay > 0:
        fh.write('\n')
        fh.write('    delay => {},\n'.format(delay_str))
    fh.write('\n')
    fh.write('    m_axis_tvalid => {},\n'.format(m_tvalid_str))
    fh.write('    m_axis_tdata => {},\n'.format(m_tdata_str))
    if tlast:
        fh.write('    m_axis_tlast => {},\n'.format(m_tlast_str))
    if tuser_width:
        fh.write('    m_axis_tuser => {},\n'.format(m_tuser_str))
    fh.write('    m_axis_tready => {}\n'.format(m_tready_str))
    fh.write(');\n')

def gen_rom(path, fi_obj, rom_type='sp', rom_style='block', prefix=''):
    """
        Generates single, dual, and true dual port rams.
    """
    path = ret_valid_path(path)
    depth = fi_obj.len
    width = fi_obj.qvec[0]
    bin_vec = fi_obj.bin
    addr_bits = int(np.ceil(np.log2(depth)))
    rom_depth = 2 ** addr_bits
    addr_msb = addr_bits - 1
    data_msb = width - 1

    def gen_port(fh, port='a'):
        fh.write('port_proc_{} : process(clk)\n'.format(port))
        fh.write('begin\n')
        # old data is presented on the output port
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        addr{}_d <= addr{};\n'.format(port, port))
        fh.write('        rom_pipe{} <= rom(to_integer(unsigned(addr{}_d)));\n'.format(port, port))
        fh.write('        do{}_d <= rom_pipe{};\n'.format(port, port))
        fh.write('    end if;\n')
        fh.write('end process port_proc_{};\n'.format(port))

        # New data is made available immediately on the output port.

    file_name = '{}{}_rom.vhd'.format(prefix, rom_type)
    file_name = os.path.join(path, file_name)
    module_name = ret_module_name(file_name)
    with open(file_name, 'w') as fh:
        fh.write('\n')
        fh.write('--*****************************************************************************\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : {}.v\n'.format(module_name))
        fh.write('-- Description : Implements a single port RAM with block ram. The ram is a fully\n')
        fh.write('--               pipelined implementation -- 3 clock cycles from new read address\n')
        fh.write('--               to new data                                                     \n')
        fh.write('--\n')
        # print_header(fh)
        fh.write('--\n')
        fh.write('--****************************************************************************\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('package {}_cmp is\n'.format(module_name))
        fh.write('    component {}\n'.format(module_name))
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        if rom_type == 'sp':
            fh.write('\n')
            fh.write('            addra : in std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('            doa : out std_logic_vector({} downto 0)\n'.format(data_msb))

        elif rom_type == 'dp':
            fh.write('            addra : in std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('            addrb : in std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('            doa : out std_logic_vector({} downto 0);\n'.format(data_msb))
            fh.write('            dob : out std_logic_vector({} downto 0)\n'.format(data_msb))
        fh.write('        );\n')
        fh.write('    end component;\n')
        fh.write('end package {}_cmp;\n'.format(module_name))
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        if rom_type == 'sp':
            fh.write('\n')
            fh.write('        addra : in std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('        doa : out std_logic_vector({} downto 0)\n'.format(data_msb))

        elif rom_type == 'dp':
            fh.write('        addra : in std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('        addrb : in std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('        doa : out std_logic_vector({} downto 0);\n'.format(data_msb))
            fh.write('        dob : out std_logic_vector({} downto 0)\n'.format(data_msb))
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n'.format(module_name))
        fh.write('\n')
        fh.write('type rom_type is array (0 to {}) of std_logic_vector({} downto 0);\n'.format(rom_depth -1, data_msb))
        fh.write('signal rom : rom_type := (\"{}\",\n'.format(bin_vec[0]))
        for i in range(1, rom_depth):
            if i < depth:
                fh.write('                          \"{}\"'.format(bin_vec[i]))
            else:
                fh.write('                          \"{}\"'.format('0' * width))

            if i == (rom_depth - 1):
                fh.write(');\n')
            else:
                fh.write(',\n')

        fh.write('\n')
        fh.write('attribute rom_style : string;\n')
        fh.write('attribute rom_style of rom : signal is \"{}\";\n'.format(rom_style))


        # fh.write('(* rom_style = \"{}\" *) reg [{}:0] rom [{}:0];\n'.format(rom_style, data_msb, rom_depth - 1))
        if rom_type == 'sp':
            fh.write('signal addra_d : std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('signal doa_d : std_logic_vector({} downto 0);\n'.format(data_msb))
            fh.write('signal rom_pipea : std_logic_vector({} downto 0);\n'.format(data_msb))

        if rom_type == 'dp':
            # fh.write('(* ram_style = \"{}\" *) reg [{}:0] ramb [{}:0];\n'.format(ram_style, data_msb, depth - 1))
            fh.write('signal addra_d : std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('signal addrb_d : std_logic_vector({} downto 0);\n'.format(addr_msb))
            fh.write('signal doa_d, dob_d : std_logic_vector({} downto 0);\n'.format(data_msb))
            fh.write('signal rom_pipea, rom_pipeb : std_logic_vector({} downto 0);\n'.format(data_msb))

        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        if rom_type == 'sp':
            fh.write('doa <= doa_d;\n')

        if rom_type == 'dp':
            fh.write('doa <= doa_d;\n')
            fh.write('dob <= dob_d;\n')

        fh.write('\n')
        # fh.write('initial\n')
        # fh.write('begin\n')
        # for i in range(rom_depth):
        #     if i < depth:
        #         fh.write('    rom[{}] = {}\'b{};\n'.format(i, width, bin_vec[i]))
        #     else:
        #         fh.write('    rom[{}] = {}\'b{};\n'.format(i, width, '0' * width))
        #         # fh.write('    $readmemb("{}", rom);\n'.format(rom_file))
        # fh.write('end\n\n')

        if rom_type == 'sp':
            gen_port(fh, 'a')

        if rom_type == 'dp':
            gen_port(fh, 'a')
            gen_port(fh, 'b')

        fh.write('\n')
        fh.write('end rtl;\n')

    return (file_name, module_name)

def gen_ram(path, ram_type='sp', memory_type='write_first', ram_style='block', fi_obj=None, prefix=''):
    """
        Generates single, dual, and true dual port rams.
    """
    path = ret_valid_path(path)
    if fi_obj is not None:
        depth = fi_obj.len
        width = fi_obj.qvec[0]
        bin_vec = fi_obj.bin
        addr_bits = int(np.ceil(np.log2(depth)))
        ram_depth = 2 ** addr_bits

    def gen_port(fh, port='a', memory_type='write_first'):
        fh.write('-- port {}\n'.format(port))
        fh.write('port_proc_{} : process(clk)\n'.format(port))
        fh.write('begin\n')
        # old data is presented on the output port
        if memory_type == 'read_first':
            fh.write('    if (we{}_d = \'1\') then\n'.format(port))
            fh.write('        ram(to_integer(unsigned(addr{}_d))) <= di{}_d;\n'.format(port, port))
            fh.write('    end if;\n')
            fh.write('    di{}_d <= di{};\n'.format(port, port))
            fh.write('    addr{}_d <= addr{};\n'.format(port, port))
            fh.write('    we{}_d <= we{};\n'.format(port, port))
            fh.write('    ram_pipe{} <= ram(addr{}_d);\n'.format(port, port))
            fh.write('    do{}_d <= ram_pipe{};\n'.format(port, port))

        # New data is made available immediately on the output port.
        if memory_type == 'write_first':
            fh.write('    if (we{}_d = \'1\') then\n'.format(port))
            fh.write('        ram(to_integer(unsigned(addr{}_d))) <= di{}_d;\n'.format(port, port))
            fh.write('        ram_pipe{} <= di{}_d;\n'.format(port, port))
            fh.write('    else\n')
            fh.write('        ram_pipe{} <= ram(to_integer(unsigned(addr{}_d)));\n'.format(port, port))
            fh.write('    end if;\n')
            fh.write('    di{}_d <= di{};\n'.format(port, port))
            fh.write('    addr{}_d <= addr{};\n'.format(port, port))
            fh.write('    we{}_d <= we{};\n'.format(port, port))
            fh.write('    do{}_d <= ram_pipe{};\n'.format(port, port))

        # the output port is not changed during a write.
        if memory_type == 'no_change':
            fh.write('    if (we{}_d = \'1\') then\n'.format(port))
            fh.write('        ram(to_integer(unsigned(addr{}_d))) <= di{}_d;\n'.format(port, port))
            fh.write('    else\n')
            fh.write('        ram_pipe{} <= ram(to_integer(unsigned(addr{}_d)));\n'.format(port, port))
            fh.write('    end if;\n')
            fh.write('    di{}_d <= di{};\n'.format(port, port))
            fh.write('    addr{}_d <= addr{};\n'.format(port, port))
            fh.write('    we{}_d <= we{};\n'.format(port, port))
            fh.write('    do{}_d <= ram_pipe{};\n'.format(port, port))

        fh.write('end process port_proc_{};\n'.format(port))

    file_name = '{}{}_{}_{}_ram.vhd'.format(prefix, ram_type, ram_style, memory_type)
    file_name = os.path.join(path, file_name)
    module_name = ret_module_name(file_name)
    with open(file_name, 'w') as fh:
        fh.write('\n')
        fh.write('--***************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : {}.v\n'.format(module_name))
        fh.write('-- Description : Implements a single port RAM with block ram. The ram is a fully\n')
        fh.write('--               pipelined implementation -- 3 clock cycles from new read address\n')
        fh.write('--               to new data                                                     \n')
        fh.write('--\n')
        # print_header(fh)
        fh.write('--\n')
        fh.write('--***************************************************************************--\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        print_intro(fh, module_name)
        fh.write('        generic\n')
        fh.write('        (\n')
        fh.write('            DATA_WIDTH : integer := 32;\n')
        fh.write('            ADDR_WIDTH : integer := 8\n')
        fh.write('        );\n')
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('\n')
        fh.write('            wea : in std_logic;\n')
        if ram_type == 'sp':
            fh.write('            addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('            di : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('            do : out std_logic_vector(DATA_WIDTH-1 downto 0)\n')
        elif ram_type == 'dp':
            fh.write('            addra : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('            addrb : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('            dia : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('            dob : out std_logic_vector(DATA_WIDTH-1 downto 0)\n')

        elif ram_type == 'tdp':
            fh.write('            web : in std_logic;\n')
            fh.write('            addra : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('            addrb : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('            dia : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('            dib : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('            doa : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('            dob : out std_logic_vector(DATA_WIDTH-1 downto 0)\n')
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    generic\n')
        fh.write('    (\n')
        fh.write('        DATA_WIDTH : integer := 32;\n')
        fh.write('        ADDR_WIDTH : integer := 8\n')
        fh.write('    );\n')
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('\n')
        fh.write('        wea : in std_logic;\n')
        if ram_type == 'sp':
            fh.write('        addr : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('        di : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('        do : out std_logic_vector(DATA_WIDTH-1 downto 0)\n')
        elif ram_type == 'dp':
            fh.write('        addra : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('        addrb : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('        dia : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('        dob : out std_logic_vector(DATA_WIDTH-1 downto 0)\n')

        elif ram_type == 'tdp':
            fh.write('        web : in std_logic;\n')
            fh.write('        addra : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('        addrb : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('        dia : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('        dib : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('        doa : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')
            fh.write('        dob : out std_logic_vector(DATA_WIDTH-1 downto 0)\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n\n'.format(module_name))
        fh.write('constant ADDR_P1 : integer := ADDR_WIDTH + 1;\n')
        fh.write('constant DATA_MSB : integer := DATA_WIDTH - 1;\n')
        fh.write('constant ADDR_MSB : integer := ADDR_WIDTH - 1;\n')
        fh.write('constant DEPTH : integer := 2 ** ADDR_WIDTH;\n\n')
        fh.write('type ram_type is array (0 to DEPTH-1) of std_logic_vector(DATA_MSB downto 0);\n')

        if fi_obj is None:
            fh.write('signal ram : ram_type := (others => (others=>\'0\'));\n')
        else:
            fh.write('signal ram : ram_type := (\"{}\",\n'.format(bin_vec[0]))
            for i in range(1, ram_depth):
                if i < depth:
                    fh.write('                          \"{}\"'.format(bin_vec[i]))
                else:
                    fh.write('                          \"{}\"'.format('0' * width))

                if i == (ram_depth - 1):
                    fh.write(');\n')
                else:
                    fh.write(',\n')
        # fh.write('(* ram_style = \"{}\" *) reg [DATA_MSB:0] ram [DEPTH-1:0];\n\n'.format(ram_style))
        fh.write('\n')
        fh.write('attribute ram_style : string;\n')
        fh.write('attribute ram_style of ram : signal is \"{}\";\n'.format(ram_style))
        fh.write('\n')
        if ram_type == 'sp':
            fh.write('signal addra_d : std_logic_vector(ADDR_MSB downto 0);\n')
            fh.write('signal wea_d : std_logic;\n')
            fh.write('signal dia_d : std_logic_vector(DATA_MSB downto 0);\n')
            fh.write('signal doa_d : std_logic_vector(DATA_MSB downto 0);\n')
            fh.write('signal ram_pipea : std_logic_vector(DATA_MSB downto 0);\n')

        if ram_type == 'dp':
            fh.write('signal addra_d : std_logic_vector(ADDR_MSB downto 0);\n')
            fh.write('signal addrb_d : std_logic_vector(ADDR_MSB downto 0);\n')
            fh.write('signal wea_d : std_logic;\n')
            fh.write('signal dia_d : std_logic_vector(DATA_MSB downto 0);\n')
            fh.write('signal dob_d : std_logic_vector(DATA_MSB downto 0);\n')
            fh.write('signal ram_pipe : std_logic_vector(DATA_MSB downto 0);\n')

        if ram_type == 'tdp':
            # fh.write('(* ram_style = \"{}\" *) reg [DATA_MSB:0] ramb [DEPTH-1:0];\n'.format(ram_style))
            fh.write('signal wea_d : std_logic;\n')
            fh.write('signal web_d : std_logic;\n')
            fh.write('signal addra_d : std_logic_vector(ADDR_MSB downto 0);\n')
            fh.write('signal addrb_d : std_logic_vector(ADDR_MSB downto 0);\n')
            fh.write('signal dia_d, dib_d : std_logic_vector(DATA_MSB downto 0);\n')
            fh.write('signal doa_d, dob_d : std_logic_vector(DATA_MSB downto 0);\n')
            fh.write('signal ram_pipea, ram_pipeb : std_logic_vector(DATA_MSB downto 0);\n')

        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        if ram_type == 'sp':
            fh.write('do <= doa_d;\n')

        if ram_type == 'dp':
            fh.write('dob <= dob_d;\n')

        if ram_type == 'tdp':
            fh.write('doa <= doa_d;\n')
            fh.write('dob <= dob_d;\n')

        fh.write('\n')
        if ram_type == 'sp':
            gen_port(fh, 'a', memory_type=memory_type)

        if ram_type == 'dp':
            fh.write('-- port a\n')
            fh.write('port_a_proc : process(clk)\n')
            fh.write('begin\n')
            fh.write('    if (rising_edge(clk)) then\n')
            fh.write('        if (wea_d = \'1\') then\n')
            fh.write('            ram(to_integer(unsigned(addra_d))) <= dia_d;\n')
            fh.write('        end if;\n')
            fh.write('        dia_d <= dia;\n')
            fh.write('        addra_d <= addra;\n')
            fh.write('        wea_d <= wea;\n')
            fh.write('    end if;\n')
            fh.write('end process port_a_proc;\n\n')
            fh.write('-- port b\n')
            fh.write('port_b_proc : process(clk)\n')
            fh.write('begin\n')
            fh.write('    if (rising_edge(clk)) then\n')
            fh.write('        addrb_d <= addrb;\n')
            fh.write('        ram_pipe <= ram(to_integer(unsigned(addrb_d)));\n')
            fh.write('        dob_d <= ram_pipe;\n')
            fh.write('    end if;\n')
            fh.write('end process port_b_proc;\n')

        if ram_type == 'tdp':
            gen_port(fh, 'a', memory_type=memory_type)
            gen_port(fh, 'b', memory_type=memory_type)

        fh.write('\n')
        fh.write('end rtl;\n')
        return module_name


def gen_axi_fifo(path, tuser_width=0, tlast=False, almost_full=False, almost_empty=False, count=False,
                 max_delay=0, ram_style='block', prefix=''):

    assert(path is not None), 'User must specify Path'
    path = ret_valid_path(path)

    if tuser_width > 0:
        tuser_msb = tuser_width - 1

    id_val = 0
    if tlast:
        id_val += 1
    if almost_full:
        id_val += 2
    if almost_empty:
        id_val += 4
    if count:
        id_val += 8
    if ram_style == 'distributed':
        id_val += 16
    if tuser_width:
        id_val += 32

    msb = fp_utils.ret_num_bitsU(id_val)
    if max_delay:
        id_val += 64   # (max_delay << msb)

    # width_msb = width - 1
    file_name = path + '{}axi_fifo_{}.vhd'.format(prefix, id_val)
    low_logic = almost_empty

    if max_delay > 0:
        delay_bits = fp_utils.ret_num_bitsU(max_delay)  #int(np.ceil(np.log2(max_delay)))
        delay_msb = delay_bits - 1

    out_cnt = count or (almost_full) or (almost_empty)
    module_name = ret_module_name(file_name)
    with open(file_name, "w") as fh:
        fh.write('--***************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : PJV\n')
        fh.write('-- File        : {}\n'.format(module_name))
        fh.write('-- Description : Generates FIFO with AXI interface. \n')
        fh.write('--\n')
        # print_header(fh)
        fh.write('--                Latency = 3.\n')
        fh.write('--\n')
        fh.write('--***************************************************************************--\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        print_intro(fh, module_name)
        fh.write('        generic\n')
        fh.write('        (\n')
        fh.write('            DATA_WIDTH : integer := 32;\n')
        if almost_full:
            fh.write('            ALMOST_FULL_THRESH : integer := 16;\n')
        if almost_empty:
            fh.write('            ALMOST_EMPTY_THRESH : integer := 8;\n')
        if tuser_width:
            fh.write('            TUSER_WIDTH : integer := 8;\n')
        fh.write('            ADDR_WIDTH : integer := 8\n')
        fh.write('        );\n')
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('    \n')
        fh.write('            s_axis_tvalid : in std_logic;\n')
        fh.write('            s_axis_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        if tlast is True:
            fh.write('            s_axis_tlast : in std_logic;\n')
        if tuser_width > 0:
            fh.write('            s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')

        fh.write('            s_axis_tready : out std_logic;\n')
        fh.write('\n')
        if max_delay > 0:
            fh.write('            delay : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
        if almost_full:
            fh.write('            almost_full : out std_logic;\n')
        if almost_empty:
            fh.write('            almost_empty : out std_logic;\n')
        if max_delay > 0 or almost_full or almost_empty:
            fh.write('\n')
        if count:
            fh.write('            data_cnt : out std_logic_vector(ADDR_WIDTH downto 0);\n')
            fh.write('\n')
        fh.write('            m_axis_tvalid : out std_logic;\n')
        fh.write('            m_axis_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')

        if tlast is True:
            fh.write('            m_axis_tlast : out std_logic;\n')
        if tuser_width > 0:
            fh.write('            m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('            m_axis_tready : in std_logic\n')
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    generic\n')
        fh.write('    (\n')
        fh.write('        DATA_WIDTH : integer :=32;\n')
        if almost_full:
            fh.write('        ALMOST_FULL_THRESH : integer := 16;\n')
        if almost_empty:
            fh.write('        ALMOST_EMPTY_THRESH : integer := 8;\n')
        if tuser_width:
            fh.write('        TUSER_WIDTH : integer := 8;\n')
        fh.write('        ADDR_WIDTH : integer := 8\n')
        fh.write('    );\n')
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('        s_axis_tvalid : in std_logic;\n')
        fh.write('        s_axis_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        if tlast is True:
            fh.write('        s_axis_tlast : in std_logic;\n')
        if tuser_width > 0:
            fh.write('        s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')

        fh.write('        s_axis_tready : out std_logic;\n')
        if max_delay > 0:
            fh.write('        delay : in std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
        fh.write('\n')
        if almost_full:
            fh.write('        almost_full : out std_logic;\n')
        if almost_empty:
            fh.write('        almost_empty : out std_logic;\n')
        if count:
            fh.write('        data_cnt : out std_logic_vector(ADDR_WIDTH downto 0);\n')
        fh.write('\n')
        fh.write('        m_axis_tvalid : out std_logic;\n')
        fh.write('        m_axis_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')

        if tlast is True:
            fh.write('        m_axis_tlast : out std_logic;\n')
        if tuser_width > 0:
            fh.write('        m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('        m_axis_tready : in std_logic\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n'.format(module_name))
        fh.write('\n')
        fh.write('constant ADDR_P1  : integer := ADDR_WIDTH + 1;\n')
        if tuser_width > 0:
            if tlast is False:
                fh.write('constant FIFO_WIDTH : integer := DATA_WIDTH + TUSER_WIDTH;\n')
            else:
                fh.write('constant FIFO_WIDTH : integer := DATA_WIDTH + TUSER_WIDTH + 1;\n')
        else:
            if tlast:
                fh.write('constant FIFO_WIDTH : integer := DATA_WIDTH + 1;\n')
            else:
                fh.write('constant FIFO_WIDTH : integer := DATA_WIDTH;\n')

        fh.write('constant FIFO_MSB : integer := FIFO_WIDTH - 1;\n')
        fh.write('constant ADDR_MSB : integer := ADDR_WIDTH - 1;\n')
        fh.write('constant DEPTH : integer := 2 ** ADDR_WIDTH;\n')
        fh.write('constant PAD : std_logic_vector(ADDR_WIDTH - 1 downto 0) := (others => \'0\');\n')
        if almost_full:
            fh.write('constant high_thresh : unsigned(ADDR_WIDTH downto 0) := to_unsigned(ALMOST_FULL_THRESH, ADDR_P1);\n')

        if low_logic:
            fh.write('constant low_thresh : unsigned(ADDR_WIDTH downto 0) := to_unsigned(ALMOST_EMPTY_THRESH, ADDR_P1);\n')

        fh.write('\n')
        if out_cnt:
            fh.write('signal data_cnt_s : unsigned(ADDR_WIDTH downto 0) := (others => \'0\');\n')

        if almost_full:
            fh.write('signal high_compare : unsigned(ADDR_P1 downto 0);\n')

        if low_logic:
            fh.write('signal low_compare : unsigned(ADDR_P1 downto 0);\n')

        fh.write('signal wr_ptr, next_wr_ptr : unsigned(ADDR_WIDTH downto 0)  := (others => \'0\');\n')
        fh.write('signal wr_addr, next_wr_addr : unsigned(ADDR_WIDTH downto 0)  := (others => \'0\');\n')
        fh.write('signal rd_ptr, next_rd_ptr : unsigned(ADDR_WIDTH downto 0)  := (others => \'0\');\n')
        fh.write('\n')
        # need attribute here.
        fh.write('type ram_type is array (DEPTH-1 downto 0) of std_logic_vector(FIFO_MSB downto 0);\n')
        fh.write('signal ram : ram_type := (others => (others=>\'0\'));\n')
        fh.write('\n')
        fh.write('attribute ram_style : string;\n')
        fh.write('attribute ram_style of ram : signal is \"{}\";\n'.format(ram_style))
        fh.write('\n')
        # fh.write('(* ram_style = "{}" *) reg [FIFO_MSB:0] ram [DEPTH-1:0];\n'.format(ram_style))
        fh.write('signal wr_data : std_logic_vector(FIFO_MSB downto 0);\n')
        fh.write('signal wr : std_logic;\n')
        fh.write('signal rd : std_logic;\n')
        fh.write('signal occ_reg, next_occ_reg : std_logic_vector(1 downto 0);\n')
        fh.write('signal data_d0, data_d1, next_data_d0, next_data_d1 : std_logic_vector(FIFO_MSB downto 0);\n')
        fh.write('\n')
        if max_delay > 0:
            fh.write('signal delay_d1, next_delay_d1 : std_logic_vector(ADDR_WIDTH-1 downto 0);\n')
            fh.write('signal add_delay : std_logic;\n')
            fh.write('signal delay_s : unsigned(ADDR_WIDTH downto 0);\n')
        fh.write('-- full when first MSB different but rest same\n')
        fh.write('signal empty, full : std_logic;\n')
        # tup_val = (addr_width, addr_width, addr_msb, addr_msb)
        fh.write('\n')
        fh.write('begin\n')
        fh.write('-- control signals\n')
        if max_delay > 0:
            fh.write('full <= \'1\' when ((wr_addr(ADDR_WIDTH) /= rd_ptr(ADDR_WIDTH)) and (wr_addr(ADDR_MSB downto 0) = rd_ptr(ADDR_MSB downto 0))) else \'0\';\n')
        else:
            fh.write('full <= \'1\' when ((wr_ptr(ADDR_WIDTH) /= rd_ptr(ADDR_WIDTH)) and (wr_ptr(ADDR_MSB downto 0) = rd_ptr(ADDR_MSB downto 0))) else \'0\';\n')

        fh.write('\n')
        fh.write('s_axis_tready <= not full;\n')
        fh.write('m_axis_tvalid <= occ_reg(1);\n')
        fh.write('-- empty when pointers match exactly\n')
        fh.write('empty <= \'1\' when (wr_ptr = rd_ptr) else \'0\';\n')
        fh.write('\n')

        if tuser_width == 0:
            if tlast is False:
                fh.write('wr_data <= s_axis_tdata;\n')
                fh.write('m_axis_tdata <= data_d1;\n')
            else:
                fh.write('wr_data <= s_axis_tlast & s_axis_tdata;\n')
                fh.write('m_axis_tdata <= data_d1(DATA_WIDTH-1 downto 0);\n')
                fh.write('m_axis_tlast <= data_d1(FIFO_MSB);\n')
        else:
            if tlast is False:
                fh.write('wr_data <= s_axis_tuser & s_axis_tdata;\n')
                fh.write('m_axis_tdata <= data_d1(DATA_WIDTH-1 downto 0);\n')
                fh.write('m_axis_tuser <= data_d1(FIFO_MSB downto DATA_WIDTH);\n')
            else:
                fh.write('wr_data <= s_axis_tlast & s_axis_tuser & s_axis_tdata;\n')
                fh.write('m_axis_tdata <= data_d1(DATA_WIDTH-1 downto 0);\n')
                fh.write('m_axis_tuser <= data_d1(FIFO_MSB-1 downto DATA_WIDTH);\n')
                fh.write('m_axis_tlast <= data_d1(FIFO_MSB);\n')
        # fh.write('assign {m_axis_tlast, m_axis_tuser, m_axis_tdata} = output_data;\n')
        if almost_full:
            fh.write('almost_full <= high_compare(ADDR_WIDTH) when (empty = \'0\') else \'0\';\n')
        if low_logic:
            fh.write('almost_empty <= low_compare(ADDR_WIDTH) when (full = \'0\') else \'0\';\n')

        if count:
            fh.write('data_cnt <= std_logic_vector(data_cnt_s);\n')

        if max_delay > 0:
            fh.write('add_delay <= \'1\' when (delay /= delay_d1) else \'0\';\n')
            fh.write('delay_s <= \'0\' & unsigned(delay);\n')
        fh.write('\n')
        fh.write('-- Write logic\n')
        fh.write('write_proc:\n')
        fh.write('process(wr_ptr, wr_addr,')
        if max_delay > 0:
            fh.write(' delay_d1,')
            fh.write(' add_delay,')
            fh.write(' delay_s,')
        fh.write(' full, s_axis_tvalid)\n')
        fh.write('begin\n')
        fh.write('    wr <= \'0\';\n')
        fh.write('    next_wr_ptr <= wr_ptr;\n')
        fh.write('    next_wr_addr <= wr_addr;\n')
        if max_delay > 0:
            fh.write('    next_delay_d1 <= delay_d1;\n')
        fh.write('\n')
        fh.write('    if (s_axis_tvalid = \'1\') then\n')
        fh.write('        -- input data valid\n')
        fh.write('        if (full = \'0\') then\n')
        fh.write('            -- not full, perform write\n')
        fh.write('            wr <= \'1\';\n')
        fh.write('            next_wr_ptr <= wr_ptr + 1;\n')
        if max_delay == 0:
            fh.write('            next_wr_addr <= wr_addr + 1;\n')
        else:
            fh.write('            if (add_delay = \'1\') then\n')
            fh.write('                next_wr_addr <= wr_ptr + delay_s + 1;\n')  #.format(delay_bits - 1))
            fh.write('                next_delay_d1 <= delay;\n')
            fh.write('            else\n')
            fh.write('                next_wr_addr <= wr_addr + 1;\n')
            fh.write('            end if;\n')

        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')

        if out_cnt:
            fh.write('-- Data Cnt Logic\n')
            fh.write('data_cnt_proc:\n')
            fh.write('process(clk)\n')
            fh.write('begin\n')
            fh.write('    if (rising_edge(clk)) then\n')
            fh.write('        data_cnt_s <= next_wr_ptr - next_rd_ptr + occ_reg_add;\n')
            if almost_full:
                fh.write('        high_compare <= resize(high_thresh - data_cnt_s, ADDR_P1 + 1);\n')
            if low_logic:
                fh.write('        low_compare <= resize(data_cnt_s - low_thresh, ADDR_P1 + 1);\n')
            fh.write('    end if;\n')
            fh.write('end process;\n\n')

        fh.write('-- main clock process\n')
        fh.write('main_clk_proc:\n')
        fh.write('process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        fh.write('            wr_ptr <= (others => \'0\');\n')
        if max_delay:
            fh.write('            wr_addr <= delay_s;\n')
        else:
            fh.write('            wr_addr <= (others => \'0\');\n')
        fh.write('            occ_reg <= (others => \'0\');\n')
        fh.write('            data_d0 <= (others => \'0\');\n')
        fh.write('            data_d1 <= (others => \'0\');\n')
        if max_delay > 0:
            fh.write('            delay_d1 <= (others => \'0\');\n')
        fh.write('        else\n')
        fh.write('            wr_ptr <= next_wr_ptr;\n')
        fh.write('            wr_addr <= next_wr_addr;\n')
        fh.write('            occ_reg <= next_occ_reg;\n')
        fh.write('            data_d0 <= next_data_d0;\n')
        fh.write('            data_d1 <= next_data_d1;\n')
        if max_delay > 0:
            fh.write('            delay_d1 <= next_delay_d1;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')

        fh.write('-- write process\n')
        fh.write('wr_clk_proc:\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (wr = \'1\') then\n')
        fh.write('            ram(to_integer(wr_addr(ADDR_MSB downto 0))) <= wr_data;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('-- Read logic\n')
        fh.write('read_proc:\n')
        fh.write('process(occ_reg, rd_ptr, data_d0, data_d1, empty, rd, m_axis_tready, ram)\n')
        fh.write('begin\n')
        fh.write('    rd <= \'0\';\n')
        fh.write('    next_rd_ptr <= rd_ptr;\n')
        fh.write('    next_occ_reg(0) <= occ_reg(0);\n')
        fh.write('    next_occ_reg(1) <= occ_reg(1);\n')
        fh.write('    next_data_d0 <= data_d0;\n')
        fh.write('    next_data_d1 <= data_d1;\n')
        fh.write('    if (occ_reg /= \"11\" or m_axis_tready = \'1\') then\n')
        fh.write('        -- output data not valid OR currently being transferred\n')
        fh.write('        if (empty = \'0\') then\n')
        fh.write('            -- not empty, perform read\n')
        fh.write('            rd <= \'1\';\n')
        fh.write('            next_rd_ptr <= rd_ptr + 1;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('    if (rd = \'1\') then\n')
        fh.write('        next_occ_reg(0) <= \'1\';\n')
        fh.write('    elsif (m_axis_tready = \'1\' or occ_reg(1) = \'0\') then\n')
        fh.write('        next_occ_reg(0) <= \'0\';\n')
        fh.write('    end if;\n')
        fh.write('    if (m_axis_tready = \'1\' or occ_reg(1) = \'0\') then\n')
        fh.write('        next_occ_reg(1) <= occ_reg(0);\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('    if (rd = \'1\') then\n')
        fh.write('        next_data_d0 <= ram(to_integer(rd_ptr(ADDR_MSB downto 0)));\n')
        fh.write('    end if;\n')
        fh.write('    if (m_axis_tready = \'1\' or occ_reg(1) = \'0\') then\n')
        fh.write('        next_data_d1 <= data_d0;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('-- read clock process\n')
        fh.write('rd_clk_proc:\n')
        fh.write('process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        fh.write('            rd_ptr <= (others => \'0\');\n')
        fh.write('        else\n')
        fh.write('            rd_ptr <= next_rd_ptr;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('end rtl;\n')

    return (file_name, module_name)

def gen_adder(path=None, a_width=16, b_width=16, subtract=False, signeda=False, signedb=False, tot_latency=None, accum=False):
    """
        Generates pipelined adder logic.  It is not an entire module, just the
        necessary logic for the adder

    """
    assert(path is not None), 'User must specify Path'
    path = ret_valid_path(path)
    a_width = ret_mult_eight(a_width)
    b_width = ret_mult_eight(b_width)
    max_width = np.max((a_width, b_width)) + 1

    out_msb = max_width - 1
    num_clocks = int(np.ceil((max_width - 1) / 8.))

    if accum is False:
        if subtract is True:
            ar_str = '-'
            mod_str = 'sub_{}_{}_l{}'.format(a_width, b_width, num_clocks)
        else:
            ar_str = '+'
            mod_str = 'add_{}_{}_l{}'.format(a_width, b_width, num_clocks)
    else:
        ar_str = '+'
        mod_str = 'accum_{}_l{}'.format(a_width, num_clocks)

    if path is not None:
        file_name = path + '/' + mod_str + '.vhd'
    else:
        file_name = './' + mod_str + '.vhd'

    pdelay = adder_pipeline(max_width)
    if tot_latency is not None:
        pad = tot_latency - pdelay
        if pad < 0:
            pad = 0
            tot_latency = pdelay
    else:
        tot_latency = pdelay
        pad = 0

    module_name = ret_module_name(file_name)
    with open(file_name, "w") as fh:
        fh.write('--***************************************************************************--\n')
        fh.write('--')
        fh.write('-- Author      : Python Generated\n')
        fh.write('-- File        : {}\n'.format(module_name))
        fh.write('-- Description : Implements a fully pipelined adder.\n')
        fh.write('--\n')
        fh.write('--\n')
        fh.write('--\n')
        fh.write('-- LICENSE     : SEE LICENSE FILE AGREEMENT,\n')
        fh.write('--\n')
        fh.write('--\n')
        print_header(fh)
        fh.write('--\n')
        fh.write('--***************************************************************************--\n')

        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        print_intro(fh, module_name)
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        # fh.write('    input sync_reset,\n')
        fh.write('            valid_i : in std_logic;\n')
        fh.write('            a : in std_logic_vector({} downto 0);\n'.format(a_width - 1))
        if accum is False:
            fh.write('            b : in std_logic_vector({} downto 0);\n'.format(b_width - 1))
        fh.write('            valid_o : out std_logic;\n')
        fh.write('            c : out std_logic_vector({} downto 0)\n'.format(out_msb))
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        print_libraries(fh)
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        # fh.write('    input sync_reset,\n')
        fh.write('        valid_i : in std_logic;\n')
        fh.write('        a : in std_logic_vector({} downto 0);\n'.format(a_width - 1))
        if accum is False:
            fh.write('        b : in std_logic_vector({} downto 0);\n'.format(b_width - 1))
            fh.write('        valid_o : out std_logic;\n')
            fh.write('        c : out std_logic_vector({} downto 0)\n'.format(out_msb))
        fh.write('    );\n')
        fh.write('end {};'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n\n'.format(module_name))
        # create partial adder registers.
        for ii in range(num_clocks):
            if ii == 0:
                fh.write('signal padd_{} : unsigned(8 downto 0);\n'.format(ii))
            else:
                fh.write('signal padd_{} : unsigned(9 downto 0);\n'.format(ii))

        for ii in range(num_clocks):
            for jj in range(num_clocks - ii - 1):
                if ii == 0:
                    fh.write('signal padd_delay{}_{} : unsigned(7 downto 0);\n'.format(ii, jj))
                else:
                    fh.write('signal padd_delay{}_{} : unsigned(8 downto 0);\n'.format(ii, jj))


        # create input register delays.
        if num_clocks > 1:
            for ii in range(1, num_clocks):
                fh.write('signal adelay_{} : unsigned({} downto 0);\n'.format(ii - 1, a_width - 1))
                if accum is False:
                    fh.write('signal bdelay_{} : unsigned({} downto 0);\n'.format(ii - 1, b_width - 1))

            fh.write('signal valid_d : std_logic_vector({} downto 0);\n'.format(num_clocks - 1))
        else:
            fh.write('signal valid_d : std_logic;\n')

        fh.write('begin\n')
        if num_clocks > 1:
            fh.write('valid_o <= valid_d({});\n'.format(num_clocks - 1))
        else:
            fh.write('valid_o <= valid_d;\n')

        # lidx = np.min((num_clocks * 8 - 1, max_width - 1)) - (num_clocks - 1) * 8
        str_val = 'c <= std_logic_vector('
        for ii in reversed(range(num_clocks)):
            if ii == num_clocks - 1:
                str_val += 'padd_{}(9 downto 1)'.format(ii)
            elif ii == 0:
                str_val += ' & padd_delay0_{}(7 downto 0)'.format(num_clocks - 2)
            else:
                str_val += ' & padd_delay{}_{}(8 downto 1)'.format(ii, num_clocks - ii - 2)

        str_val += ');\n'
        fh.write('{}'.format(str_val))
        fh.write('\n')
        fh.write('-- main clock process\n')
        fh.write('main_clk_proc:\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        for ii in range(num_clocks):
            if num_clocks == 1:
                fh.write('        valid_d <= valid_i;\n')
            else:
                if ii == 0:
                    fh.write('        valid_d({}) <= valid_i;\n'.format(ii))
                else:
                    fh.write('        valid_d({}) <= valid_d({});\n'.format(ii, ii - 1))
        for ii in range(num_clocks):
            idx = 8 if (ii == 1) else 9
            if ii == 0:
                if accum is True:
                    fh.write('        padd_{} <= unsigned(\'0\' & a(7 downto 0)) {} padd_0;\n'.format(ii, ar_str))
                else:
                    fh.write('        padd_{} <= unsigned(\'0\' & a(7 downto 0)) {} unsigned(\'0\' & b(7 downto 0));\n'.format(ii, ar_str))
            elif ii == (num_clocks - 1):
                lidxa = a_width - 1
                ridx = (num_clocks - 1) * 8
                lidxb = b_width - 1
                prev = num_clocks - 2

                if signeda is True:
                    pada = 'adelay_{}({})'.format(pdelay - 2, lidxa)
                else:
                    pada = '\'0\''

                if accum is False:
                    if signedb is True:
                        padb = 'bdelay_{}({})'.format(pdelay - 2, lidxb)
                    else:
                        padb = '\'0\''
                else:
                    padb = 'padd_{}({})'.format(ii, idx)

                stra = 'adelay_{}({} downto {})'.format(prev, lidxa, ridx)
                if subtract:
                    strac = '\'0\''
                else:
                    strac = 'padd_{}({})'.format(prev, idx)
                if accum is False:
                    strb = 'bdelay_{}({} downto {})'.format(prev, lidxb, ridx)
                    strad = 'padd_{}({})'.format(prev, idx)
                    t_val = (ii, pada, stra, strac, ar_str, padb, strb, strad)
                    fh.write('        padd_{} <= unsigned({} & {} & {}) {} unsigned({} & {} & {});\n'.format(*t_val))  #analysis:ignore
                else:
                    strb = 'padd_{}'.format(ii)
                    t_val = (ii, pada, stra, strac, ar_str, strb)
                    fh.write('        padd_{} <= unsigned({} & {} & {}) {} unsigned({} & {});\n'.format(*t_val))  #analysis:ignore
            else:
                lidx = ii * 8 + 7
                ridx = ii * 8
                prev = ii - 1
                if accum is True:
                    strb = 'padd_{}'.format(ii)
                    t_val = (ii, prev, lidx, ridx, ar_str, strb)
                    fh.write('        padd_{} <= \'0\' & adelay_{}({} downto {}) & \'0\' {} {};\n'.format(*t_val))
                else:
                    if subtract:
                        t_val = (ii, prev, lidx, ridx, ar_str, prev, lidx, ridx, prev, idx)
                        fh.write('        padd_{} <= unsigned(\'0\' & adelay_{}({} downto {}) & \'0\') {} unsigned(\'0\' & bdelay_{}({} downto {}) & padd_{}({}));\n'.format(*t_val))
                    else:
                        t_val = (ii, prev, lidx, ridx, prev, idx, ar_str, prev, lidx, ridx, prev, idx)
                        fh.write('        padd_{} <= unsigned(\'0\' & adelay_{}({} downto {}) & padd_{}({})) {} unsigned(\'0\' & bdelay_{}({} downto {}) & padd_{}({}));\n'.format(*t_val))  #analysis:ignore

        for ii in range(num_clocks - 1):
            if ii == 0:
                fh.write('        adelay_{} <= unsigned(a);\n'.format(ii))
                if accum is False:
                    fh.write('        bdelay_{} <= unsigned(b);\n'.format(ii))
            else:
                fh.write('        adelay_{} <= adelay_{};\n'.format(ii, ii - 1))
                if accum is False:
                    fh.write('        bdelay_{} <= bdelay_{};\n'.format(ii, ii - 1))

        for ii in range(num_clocks - 1):
            for jj in range(num_clocks - ii - 1):
                if jj == 0:
                    if ii == 0:
                        fh.write('        padd_delay{}_{} <= padd_{}(7 downto 0);\n'.format(ii, jj, ii))
                    else:
                        fh.write('        padd_delay{}_{} <= padd_{}(8 downto 0);\n'.format(ii, jj, ii))
                    # fh.write('        padd_delay{}_{} <= padd_{};\n'.format(ii, jj, ii))
                else:
                    fh.write('        padd_delay{}_{} <= padd_delay{}_{};\n'.format(ii, jj, ii, jj - 1))

        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('end rtl;\n')

    return (mod_str, num_clocks)


def gen_aligned_cnt(path, cnt_width=16, tuser_width=0, tlast=False, incr=1, tot_latency=None, start_sig=False,
                    cycle=False, upper_cnt=False, prefix='', dwn_cnt=False, load=False, dport=True, startup=True,
                    almost_full_thresh=None, fifo_addr_width=None, use_af=False):
    """
        Generates logic to align pipelined counter with a data stream.

        ==========
        Parameters
        ==========

            cnt_width : counter width.
            tot_latency : pads logic with additional pipelining for a total latency = tot_latency.  Note is tot_latency
                            is < minimum pipelining, then tot_latency = minimum pipelining.
            start_sig : start signal used to gate logic.  Logic will not drive the output until a start signal is
                        received.
            startup : Option ensure that data is not valid until valid data is read from the internal FIFO.
            upper_cnt : (optional) counts number of cycles through lower counter.
            cycle : indicates that the system should cycle through the last n values for each new value.  This
                    is useful for MAC base filtering.
    """

    assert(path is not None), 'User must specify Path'
    path = ret_valid_path(path)

    if dport is False:
        tlast = False
        tuser_width = 0

    cnt_width = int(np.ceil(cnt_width / 8.)) * 8
    pdelay = adder_pipeline(cnt_width)

    sig_list = ['reset_cnt', 'take_data', 'cnt_reset', 'cnt_nib0']
    for ii in range(pdelay - 2):
        sig_list.append('reset_cnt_d{}'.format(ii))
    for ii in range(pdelay - 1):
        sig_list.append('cnt_reset_{}'.format(ii))
    for ii in range(pdelay - 1):
        sig_list.append('take_d{}'.format(ii))

    id_val = 0
    if tlast:
        id_val += 1
    if start_sig:
        id_val += 2
        sig_list.append('new_cnt')
        for ii in range(pdelay - 1):
            sig_list.append('new_cnt_d{}'.format(ii))

    if upper_cnt:
        id_val += 4
    if dwn_cnt:
        id_val += 8
    if use_af is True:
        id_val += 16

    if load:
        id_val += 32
        sig_list.append('load_cnt')
        sig_list.append('load')
        sig_list.append('load_value0')
        for ii in range(pdelay - 1):
            sig_list.append('load_reg_d{}'.format(ii))

    if tuser_width > 0:
        id_val += 64
    if tot_latency:
        id_val += 128 + tot_latency

    if cycle:
        module_name = 'process_cycle_cw{}_{}'.format(cnt_width, id_val)
    else:
        module_name = 'count_cycle_cw{}_{}'.format(cnt_width, id_val)

    if len(prefix) > 0:
        module_name = prefix + '_' + module_name

    file_name = name_help(module_name, path)
    module_name = ret_module_name(file_name)

    # generate axi_fifo
    if use_af is False:
        fifo_depth = 2 ** int(np.ceil(np.log2(pdelay*2)))
        if fifo_depth < 8:
            fifo_depth = 8
        fifo_addr_width = int(np.log2(fifo_depth))
        almost_full_thresh = 2 ** fifo_addr_width - pdelay - 1
    else:
        assert(almost_full_thresh is not None), 'User must specify almost_full_thresh when using af'
        assert(fifo_addr_width is not None), 'User must specify fifo_addr_width when using af'

    (_, fifo_name) = gen_axi_fifo(path, tuser_width=tuser_width, almost_full=almost_full_thresh, ram_style='distributed', tlast=tlast)
    if tot_latency is not None:
        pad = tot_latency - pdelay
        if pad < 0:
            pad = 0
            tot_latency = pdelay
    else:
        tot_latency = pdelay
        pad = 0

    if dwn_cnt:
        roll_over_str = 'cnt_nib0(7 downto 0) = \"00000000\"'
        for jj in range(1, pdelay):
            roll_over_str = roll_over_str + ' and next_cnt_nib{}(7 downto 0) = \"00000000\"'.format(jj)

        roll_over_strs = []
        for ii in range(1, pdelay):
            temp = 'cnt_nib0_d{}(7 downto 0) = \"00000000\"'.format(ii - 1)
            for jj in range(1, pdelay):
                if jj >= ii:
                    temp = temp + ' and cnt_nib{}(7 downto 0) = \"00000000\"'.format(jj)
                else:
                    temp = temp + ' and cnt_nib{}_d{} = \"00000000\"'.format(jj, ii - jj - 1)
            roll_over_strs.append(temp)
    else:
        roll_over_str = 'cnt_nib0(7 downto 0) = mask0'
        for jj in range(1, pdelay):
            roll_over_str = roll_over_str + ' and next_cnt_nib{}(7 downto 0) = mask{}'.format(jj, jj)

        roll_over_strs = []
        for ii in range(1, pdelay):
            temp = 'cnt_nib0_d{}(7 downto 0) = mask0'.format(ii - 1)
            for jj in range(1, pdelay):
                if jj >= ii:
                    temp = temp + ' and cnt_nib{}(7 downto 0) = mask{}'.format(jj, jj)
                else:
                    temp = temp + ' and cnt_nib{}_d{} = mask{}'.format(jj, ii - jj - 1, jj)
            roll_over_strs.append(temp)

    cnt_msb = cnt_width - 1
    sig_list = ['reset_cnt', 'take_data', 'cnt_reset']
    for ii in range(pdelay):
        sig_list.append('cnt_nib{}'.format(ii))
    if startup:
        sig_list.append('startup')
    if cycle:
        sig_list.append('cycling')
    if start_sig:
        sig_list.append('new_cnt')
    if load:
        sig_list.append(['load_cnt', 'load_value0', 'load_reg_d{}'.format(jj - 1), 'loadv_d{}'.format(jj - 1)])
    if dwn_cnt:
        for i in range(pdelay):
            sig_list.append('mask{}'.format(i))

    # tuser_msb = tuser_width - 1
    remain_val = 7 - (pdelay * 8 - cnt_width)
    pad_bits = (pdelay * 8 - cnt_width)
    int_cnt_msb = cnt_msb + pad_bits
    with open(file_name, "w") as fh:
        fh.write('--***************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author : PJV\n')
        fh.write('-- File : {}\n'.format(module_name))
        fh.write('-- Description : Implement simple count / data alignment logic while optimizing pipelining.\n')
        fh.write('--                Useful for aligning data with addition of metadata\n')
        fh.write('--\n')
        # print_header(fh)
        fh.write('--\n')
        fh.write('--***************************************************************************--\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        print_intro(fh, module_name)
        if dport:
            fh.write('        generic\n')
            fh.write('        (\n')
            if tuser_width == 0:
                fh.write('            DATA_WIDTH : integer :=32\n')
            else:
                fh.write('            DATA_WIDTH : integer :=32;\n')
                fh.write('            TUSER_WIDTH : integer :=32\n')
        fh.write('        );\n')
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('            s_axis_tvalid : in std_logic;\n')
        if dport:
            fh.write('            s_axis_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        fh.write('            cnt_limit : in std_logic_vector({} downto 0);\n'.format(cnt_msb))
        if tuser_width > 0:
            fh.write('            s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('            s_axis_tlast : in std_logic;\n')
        fh.write('            s_axis_tready : out std_logic;\n')
        fh.write('\n')
        if load:
            fh.write('            load_value : in std_logic_vector({} downto 0);\n'.format(cnt_msb))
            fh.write('            load : in std_logic;\n')
        if incr > 1:
            fh.write('            incr : in std_logic_vector(7 downto 0);\n')
        if start_sig:
            fh.write('            start_sig : in std_logic;\n'.format(cnt_msb))
        if upper_cnt:
            fh.write('            uroll_over : in std_logic_vector(7 downto 0);\n')
        if use_af:
            fh.write('            af : out std_logic;\n')
        fh.write('\n')
        fh.write('            m_axis_tvalid : out std_logic;\n')
        if dport:
            fh.write('            m_axis_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        fh.write('            m_axis_final_cnt : out std_logic;\n')
        if tuser_width > 0:
            fh.write('            m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('\n')
        fh.write('            count : out std_logic_vector({} downto 0);\n'.format(cnt_msb))
        if upper_cnt:
            fh.write('            upper_cnt : out std_logic_vector(7 downto 0);\n')
        if tlast:
            fh.write('            m_axis_tlast : out std_logic;\n')
        fh.write('            m_axis_tready : in std_logic\n')
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('library work;\n')
        fh.write('use work.{}_cmp.all;\n'.format(fifo_name))
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        if dport:
            fh.write('    generic\n')
            fh.write('    (\n')
            if tuser_width == 0:
                fh.write('        DATA_WIDTH : integer :=32\n')
            else:
                fh.write('        DATA_WIDTH : integer :=32;\n')
                fh.write('        TUSER_WIDTH : integer :=32\n')
        fh.write('    );\n')
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('        s_axis_tvalid : in std_logic;\n')
        if dport:
            fh.write('        s_axis_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        if tuser_width > 0:
            fh.write('        s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('        s_axis_tlast : in std_logic;\n')
        fh.write('        s_axis_tready : out std_logic;\n')
        fh.write('\n')
        fh.write('        cnt_limit : in std_logic_vector({} downto 0);\n'.format(cnt_msb))
        if load:
            fh.write('        load_value : in std_logic_vector({} downto 0);\n'.format(cnt_msb))
            fh.write('        load : in std_logic;\n')
        if incr > 1:
            fh.write('        incr : in std_logic_vector(7 downto 0);\n')
        if start_sig:
            fh.write('        start_sig : in std_logic;\n'.format(cnt_msb))
        if upper_cnt:
            fh.write('        uroll_over : in std_logic_vector(7 downto 0);\n')
        if use_af:
            fh.write('        af : out std_logic;\n')
        fh.write('\n')
        fh.write('        m_axis_tvalid : out std_logic;\n')
        if dport:
            fh.write('        m_axis_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        fh.write('        m_axis_final_cnt : out std_logic;\n')
        if tuser_width > 0:
            fh.write('        m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('        count : out std_logic_vector({} downto 0);\n'.format(cnt_msb))
        if upper_cnt:
            fh.write('        upper_cnt : out std_logic_vector(7 downto 0);\n')
        if tlast:
            fh.write('        m_axis_tlast : out std_logic;\n')
        fh.write('        m_axis_tready : in std_logic\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n\n'.format(module_name))
        if dport:
            fh.write('constant DATA_MSB : integer := DATA_WIDTH - 1;\n')
        if tuser_width > 0:
            fh.write('constant TUSER_MSB : integer := TUSER_WIDTH - 1;\n')
        if load:
            fh.write('signal load_cnt : std_logic;\n')
            for jj in range(pdelay - 1):
                fh.write('signal load_reg_d{} : std_logic;\n'.format(jj))
            fh.write('\n')
            fh.write('signal load_value0 : unsigned({} downto 0);\n'.format(int_cnt_msb))
            for jj in range(pdelay - 1):
                fh.write('signal loadv_d{} : unsigned({} downto 0) ;\n'.format(int_cnt_msb, jj))

        if dport:
            for jj in range(tot_latency):
                fh.write('signal data_d{} : std_logic_vector(DATA_MSB downto 0);\n'.format(jj))

        if tuser_width > 0:
            for jj in range(tot_latency):
                fh.write('signal tuser_d{} : std_logic_vector(TUSER_MSB downto 0);\n'.format(jj))

        if tlast:
            fh.write('signal tlast_d : std_logic_vector({} downto 0);\n'.format(tot_latency - 1))

        if upper_cnt:
            for jj in range(tot_latency):
                fh.write('signal upper_cnt_d{} : unsigned(7 downto 0);\n'.format(jj))

        fh.write('\n')
        if startup:
            fh.write('signal startup, next_startup : std_logic;\n')
        fh.write('signal almost_full : std_logic;\n')
        fh.write('\n')

        fh.write('signal m_fifo_tvalid : std_logic;\n')
        if dport:
            fh.write('signal m_fifo_tdata : std_logic_vector(DATA_WIDTH + {} downto 0);\n'.format(cnt_width))
        else:
            fh.write('signal m_fifo_tdata : std_logic_vector({} downto 0);\n'.format(cnt_width))
        fh.write('signal m_fifo_tready : std_logic;\n')

        if tuser_width > 0:
            fh.write('signal m_fifo_tuser : std_logic_vector(TUSER_MSB downto 0);\n')

        if tlast:
            fh.write('signal m_fifo_tlast : std_logic;\n')

        gen_cnt_sigs(fh, prefix='cnt', pdelay=pdelay)
        for nn in range(pad):
            fh.write('signal count_d{}, next_count_d{} : unsigned({} downto 0);\n'.format(cnt_msb, nn, nn))

        fh.write('signal count_s : unsigned({} downto 0);\n'.format(cnt_msb))
        fh.write('signal cnt_limit_s : unsigned({} downto 0);\n'.format(cnt_msb))
        fh.write('signal reset_cnt, next_reset_cnt : std_logic;\n')
        if cycle:
            fh.write('signal cycling, next_cycling : std_logic;\n')
        for jj in range(pdelay - 2):
            fh.write('signal reset_cnt_d{} : std_logic;\n'.format(jj))
        fh.write('\n')
        for jj in range(pdelay):
            fh.write('signal mask{} : unsigned(7 downto 0);\n'.format(jj))
        fh.write('\n')
        fh.write('signal take_data, tready_s : std_logic;\n')
        fh.write('signal final_cnt, cnt_reset : std_logic;\n')
        fh.write('signal fifo_tready : std_logic;\n')
        fh.write('signal fifo_tdata : std_logic_vector(DATA_WIDTH + {} downto 0);\n'.format(cnt_width))
        for ii in range(pdelay - 1):
            fh.write('signal cnt_reset_{} : std_logic;\n'.format(ii))
        for jj in range(pdelay):
            fh.write('signal take_d{} : std_logic;\n'.format(jj))

        if start_sig:
            fh.write('signal new_cnt : std_logic;\n')
            for jj in range(pdelay - 1):
                fh.write('signal new_cnt_d{} : std_logic;\n'.format(jj))
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        if dport:
            fh.write('fifo_tdata <= final_cnt & std_logic_vector(count_s) & data_d{};\n'.format(pdelay - 1))
        else:
            fh.write('fifo_tdata <= final_cnt & std_logic_vector(count_s);\n')

        fh.write('m_axis_tvalid <= m_fifo_tvalid;\n')
        if dport:
            fh.write('m_axis_tdata <= m_fifo_tdata(DATA_MSB downto 0);\n')
        fh.write('m_fifo_tready <= m_axis_tready;\n')
        if dport:
            fh.write('m_axis_final_cnt <= m_fifo_tdata(DATA_WIDTH + {});\n'.format(cnt_width))
        else:
            fh.write('m_axis_final_cnt <= m_fifo_tdata({});\n'.format(cnt_width))
        if tlast:
            fh.write('m_axis_tlast <= m_fifo_tlast;\n')
        if tuser_width > 0:
            fh.write('m_axis_tuser <= m_fifo_tuser;\n')
        fh.write('\n')
        if use_af:
            if cycle:
                fh.write('tready_s <= \'1\' when (fifo_tready and (not cycling or cnt_reset)) else \'0\';\n'.format(pdelay - 1))  #analysis:ignore
            else:
                fh.write('tready_s <= fifo_tready;\n')

            fh.write('af <= almost_full;\n')
        else:
            if cycle:
                fh.write('tready_s <= \'1\' when (not almost_full and (not cycling or cnt_reset)) else \'0\';\n'.format(pdelay - 1))  #analysis:ignore
            else:
                fh.write('tready_s <= not almost_full;\n')

        fh.write('take_data <= s_axis_tvalid and tready_s and (not sync_reset);\n')
        fh.write('s_axis_tready <= tready_s;\n')
        fh.write('cnt_limit_s <= unsigned(cnt_limit);\n')
        cnt = startup + start_sig + load
        str_val = ''
        if cnt > 0:
            str_val = 'and '
        if cnt > 1:
            str_val += '('
        if startup:
            str_val += 'startup'
            if cnt > 1:
                str_val += ' or '

        if start_sig:
            str_val += 'start_sig'

        if load:
            if cnt > 1:
                str_val += ' or '
            str_val += 'load'

        if cnt > 1:
            str_val += ')'

        # if startup or start_sig:
        #     str_val += ')'
        if start_sig:
            fh.write('new_cnt <= (take_data {});\n'.format(str_val)) # ? \'1\' : \'0\';\n'.format(str_val))
        if load:
            fh.write('load_cnt <= (take_data {});\n'.format(str_val)) # ? \'1\' : \'0\';\n'.format(str_val))

        if upper_cnt:
            fh.write('upper_cnt <= upper_cnt_d{};\n'.format(tot_latency - 1))
        str_val = ''

        if dwn_cnt:
            zero_str = '0' * cnt_width
            fh.write('final_cnt <= \'1\' when (count_s = \"{}\") else \'0\';\n'.format(zero_str))
        else:
            fh.write('final_cnt <= \'1\' when (count_s = cnt_limit_s) else \'0\';\n')

        fh.write('\n')
        for ii, str_val in enumerate(roll_over_strs):
            fh.write('cnt_reset_{} <= \'1\' when ({}) else \'0\';\n'.format(ii, str_val))

        fh.write('cnt_reset <= \'1\' when ({}) else \'0\';\n'.format(roll_over_str))
        fh.write('\n')
        str_val = ''
        for jj in reversed(range(pdelay - 1)):
            delay_val = pdelay - 2 - jj
            str_val = str_val + ' & cnt_nib{}_d{}'.format(jj, delay_val)
        str_val = 'cnt_nib{}({} downto 0)'.format(pdelay - 1, remain_val) + str_val + ';\n'
        fh.write('count_s <= {}'.format(str_val))
        if dport:
            fh.write('count <= m_fifo_tdata((DATA_WIDTH + {}) downto DATA_WIDTH);\n'.format(cnt_msb))
        else:
            fh.write('count <= m_fifo_tdata({} downto 0);\n'.format(cnt_msb))
        fh.write('\n')
        reset_list = []
        for jj in range(pdelay):
            lidx = jj * 8 + 7
            ridx = lidx - 7
            if jj == pdelay - 1 and remain_val != 7:
                bin_str = '0' * 7 - remain_val
                fh.write('mask{} <= \"{}\" & cnt_limit_s({} downto {});\n'.format(jj, bin_str, cnt_msb, ridx))
            else:
                fh.write('mask{} <= cnt_limit_s({} downto {});\n'.format(jj, lidx, ridx))
            reset_list.append('\'0\' & mask{}'.format(jj))


        if load:
            if jj == pdelay - 1 and remain_val != 7:
                bin_str = '0' * pad_bits
                fh.write('load_value0 <= \"{}\" & load_value;\n'.format(bin_str))
            else:
                fh.write('load_value0 <= load_value;\n')
        fh.write('\n')
        sp = '            '
        fh.write('main_clk_proc : process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        fh.write('            reset_cnt <= \'0\';\n')
        if cycle:
            fh.write('            cycling <= \'0\';\n')
        if dwn_cnt:
            gen_cnt_rst(fh, prefix='cnt', pdelay=pdelay, sp=sp, reset_list=reset_list)
        else:
            gen_cnt_rst(fh, prefix='cnt', pdelay=pdelay, sp=sp)
        for jj in range(pad):
            fh.write('            count_d{} <= (others => \'0\');\n'.format(jj))
        if upper_cnt:
            logic_rst(fh, prefix='upper_cnt_d', cnt=tot_latency, sp=sp)
        if startup:
            fh.write('            startup <= \'1\';\n')
        fh.write('        else\n')
        fh.write('            reset_cnt <= next_reset_cnt;\n')
        if cycle:
            fh.write('            cycling <= next_cycling;\n')
        gen_cnt_regs(fh, 'cnt', pdelay)
        if upper_cnt:
            logic_gate(fh, prefix='upper_cnt_d', str_val='next_upper_cnt', cnt=tot_latency, sp=sp)
        if startup:
            fh.write('            startup <= next_startup;\n')
        for jj in range(pad):
            fh.write('            count_d{} <= next_count_d{};\n'.format(jj, jj))
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process main_clk_proc;\n')
        fh.write('\n')
        fh.write('\n')
        if pdelay > 0:
            fh.write('delay_proc : process(clk)\n')
            fh.write('begin\n')
            fh.write('    if (rising_edge(clk)) then\n')
            # ipdb.set_trace()
            gen_cnt_delay(fh, 'cnt', pdelay, tab='')
            if start_sig:
                for jj in range(pdelay - 1):
                    if jj == 0:
                        fh.write('        new_cnt_d0 <= new_cnt;\n')
                    else:
                        fh.write('        new_cnt_d{} <= new_cnt_d{};\n'.format(jj, jj - 1))
                fh.write('\n')
            if dport:
                fh.write('        data_d0 <= s_axis_tdata;\n')
                for jj in range(1, tot_latency):
                    fh.write('        data_d{} <= data_d{};\n'.format(jj, jj-1))

            if tuser_width > 0:
                fh.write('        tuser_d0 <= s_axis_tuser;\n')
                for jj in range(1, tot_latency):
                    fh.write('        tuser_d{} <= tuser_d{};\n'.format(jj, jj - 1))

            if tlast:
                fh.write('        tlast_d(0) <= s_axis_tlast;\n')
                for jj in range(1, tot_latency):
                    fh.write('        tlast_d({}) <= tlast_d({});\n'.format(jj, jj - 1))

            if pdelay > 2:
                fh.write('        reset_cnt_d0 <= reset_cnt;\n')
                for jj in range(1, pdelay - 2):
                    fh.write('        reset_cnt_d{} <= reset_cnt_d{};\n'.format(jj, jj - 1))

                fh.write('\n')
                if load:
                    fh.write('        load_reg_d0 <= load_cnt;\n')
                    for jj in range(1, pdelay - 1):
                        fh.write('        load_reg_d{} <= load_reg_d{};\n'.format(jj, jj - 1))
                    fh.write('\n')
                    fh.write('        loadv_d0 <= load_value0;\n')
                    for jj in range(1, pdelay - 1):
                        fh.write('        loadv_d{} <= loadv_d{};\n'.format(jj, jj - 1))
                fh.write('\n')

            for jj in range(pdelay):
                if jj == 0:
                    if cycle:
                        fh.write('        take_d0 <= take_data or cycling;\n')
                    else:
                        fh.write('        take_d0 <= take_data;\n')
                else:
                    fh.write('        take_d{} <= take_d{};\n'.format(jj, jj - 1))
            fh.write('    end if;\n')
            fh.write('end process delay_proc;')
            fh.write('\n')
        fh.write('\n')

        fh.write('async_proc : ')
        print_sens_list(fh, sig_list)
        fh.write('begin\n')
        gen_cnt_fback(fh, 'cnt', pdelay)
        fh.write('    next_reset_cnt <= reset_cnt;\n')
        if startup:
            fh.write('    next_startup <= startup;\n')
        if cycle:
            fh.write('    next_cycling <= cycling;\n')

        if cycle:
            fh.write('    if (take_data = \'1\' or cycling = \'1\') then\n')
        else:
            fh.write('    if (take_data = \'1\') then\n')

        str_val = ''
        if startup:
            str_val = ' or startup = \'1\''
        if start_sig:
            str_val = ' or new_cnt = \'1\''
        if load:
            str_val = ' or load_cnt = \'1\''
        if cycle:
            fh.write('        next_cycling <= (not cnt_reset or take_data);\n')
        fh.write('        next_reset_cnt <= cnt_reset;\n')
        fh.write('        if (cnt_reset = \'1\'{}) then\n'.format(str_val))
        if load:
            fh.write('            if (load = \'1\') then\n')
            fh.write('                next_cnt_nib0 <= load_value0(7 downto 0);\n')
            fh.write('            else\n')
            if dwn_cnt:
                fh.write('                next_cnt_nib0 <= \'0\' & mask0;\n')
            else:
                fh.write('                next_cnt_nib0 <= (others => \'0\');\n')
            fh.write('            end if;\n')

        else:
            if dwn_cnt:
                fh.write('            next_cnt_nib0 <= \'0\' & mask0;\n')
            else:
                fh.write('            next_cnt_nib0 <= (others => \'0\');\n')
        fh.write('        else\n')
        ar_str = '-' if dwn_cnt else '+'
        if incr == 1:
            fh.write('            next_cnt_nib0 <= resize(cnt_nib0(7 downto 0), 9) {} 1;\n'.format(ar_str))
        else:
            fh.write('            next_cnt_nib0 <= resize(cnt_nib0(7 downto 0), 9) {} incr;\n'.format(ar_str))
        fh.write('        end if;\n')
        if startup:
            fh.write('        next_startup <= \'0\';\n')
        fh.write('    end if;\n')
        fh.write('\n')

        for jj in range(1, pdelay):
            str_val = ''
            if start_sig:
                str_val = str_val + ' or new_cnt_d{} = \'1\''.format(jj - 1)
            if load:
                str_val += ' or load_reg_d{}'.format(jj - 1)
            if jj == 1:
                fh.write('    if (reset_cnt = \'1\'{} or cnt_reset_{} = \'1\') then\n'.format(str_val, jj - 1))
            else:
                fh.write('    if (reset_cnt_d{} = \'1\' {} or cnt_reset_{} \'1\') begin\n'.format(jj - 2, str_val, jj - 1))

            if load:
                ridx = jj * 8
                lidx = ridx + 7
                fh.write('        if (load_reg_d{} = \'1\') then\n'.format(jj - 1))
                fh.write('            next_cnt_nib{} <= loadv_d{}({} downto {});\n'.format(jj, jj - 1, lidx, ridx))
                fh.write('        else\n')
                if dwn_cnt:
                    fh.write('            next_cnt_nib{} <= \'0\' & mask{};\n'.format(jj, jj))
                else:
                    fh.write('            next_cnt_nib{} <= (others => \'0\');\n'.format(jj))
                fh.write('        end if;\n')
            else:
                if dwn_cnt:
                    fh.write('        next_cnt_nib{} <= \'0\' & mask{};\n'.format(jj, jj))
                else:
                    fh.write('        next_cnt_nib{} <= (others => \'0\');\n'.format(jj))
            fh.write('    elsif (take_d{} = \'1\') then\n'.format(jj - 1))
            ar_str = '-' if dwn_cnt else '+'
            fh.write('        next_cnt_nib{} <= resize(cnt_nib{}(7 downto 0), 9) {} ("00000000" & cnt_nib{}(8));\n'.format(jj, jj, ar_str, jj - 1))
            fh.write('    end if;\n\n')
        fh.write('end process async_proc;\n')
        sig_list = ['upper_cnt_d0', 'take_d{}'.format(pdelay - 1)]
        if start_sig:
            sig_list.append('new_cnt')
        if cycle:
            sig_list.append('cycling')

        if upper_cnt:
            fh.write('upper_cnt_proc:\n')
            print_sens_list(fh, sig_list)
            fh.write('begin\n')
            fh.write('    next_upper_cnt = upper_cnt_d0;\n')
            if start_sig:
                fh.write('    if (take_d{} = \'1\' or new_cnt = \'1\') then\n'.format(pdelay - 1))  # analysis:ignore
            elif cycle:
                fh.write('    if (take_d{} = \'1\' or cycling = \'1\') then\n'.format(pdelay - 1))
            else:
                fh.write('    if (take_d{} = \'1\') then\n'.format(pdelay - 1))
            str_val = ''
            if start_sig:
                str_val = ' or new_cnt '
            fh.write('            next_upper_cnt <= (others => \'0\');\n')
            fh.write('        if (final_cnt = \'1\'{}) then\n'.format(str_val))
            if (start_sig):
                fh.write('            if ((upper_cnt = uroll_over) or new_cnt) then\n')
            else:
                fh.write('            if (upper_cnt = uroll_over) then\n')
            fh.write('                next_upper_cnt <= (others => \'0\');\n')
            fh.write('            else\n')
            fh.write('                next_upper_cnt <= upper_cnt + 1;\n')
            fh.write('            end if;\n')
            fh.write('        end if;\n')
            fh.write('    end if;\n')
            fh.write('end process upper_cnt_proc;\n')
        fh.write('\n')
        if pad:
            fh.write('out_proc:\n')
            sig_list = ['count_s']
            for ii in range(pad - 1):
                sig_list.append('count_d{}'.format(ii))
            print_sens_list(fh, sig_list)
            fh.write('begin\n')
            fh.write('    next_count_d0 <= count_s;\n')
            for jj in range(1, pad):
                fh.write('    next_count_d{} <= count_d{};\n'.format(jj, jj - 1))
            # str_val = ''
            fh.write('end process out_proc;\n')
        fh.write('\n')
        if dport:
            data_width = 'DATA_WIDTH + {}'.format(cnt_width+1)
        else:
            data_width = '{}'.format(cnt_width+1)

        axi_fifo_inst(fh, fifo_name, inst_name='u_fifo', data_width=data_width, af_thresh=almost_full_thresh,
                      addr_width=fifo_addr_width, tuser_width=tuser_width, tlast=tlast, s_tvalid_str='take_d{}'.format(pdelay-1),
                      s_tdata_str='fifo_tdata', s_tuser_str='tuser_d{}'.format(pdelay-1), s_tlast_str='tlast_d{}'.format(pdelay-1),
                      s_tready_str='fifo_tready', almost_full_str='almost_full', m_tvalid_str='m_fifo_tvalid', m_tdata_str='m_fifo_tdata',
                      m_tuser_str='m_fifo_tuser', m_tlast_str='m_fifo_tlast', m_tready_str='m_fifo_tready')
        fh.write('\n')
        fh.write('end rtl;\n')

    return module_name, fifo_name

def gen_pipe_logic(input_width, logic_function='xor', file_path=None):
    """
        Function generates Verilog module of a fully pipelined logic
        function.
        The inputs should be a single concatenated signal.

        ==========
        Parameters
        ==========

            * input_width (int)
                input vector width
            * logic_function (str)
                Logic operator to be fully pipelined.
            * file_path (str)
                Default current working directory.
        =======
        Returns
        =======

            Verilog file that implements a fully pipelined multiplexer.

    """

    file_name = ('pipe_%s_%d.v' % (logic_function, input_width))
    if (file_path is None):
        file_path = os.getcwd()

    file_name2 = os.path.join(file_path, file_name)

    num_stages = fp_utils.nextpow2(input_width)

    offset = file_name.rfind('.')
    module_name = file_name[:offset]

    gates = []
    rem_gates = []
    num_bits = copy.copy(input_width)
    for stage in range(num_stages):
        temp = int(np.ceil(num_bits / 2.))
        rem_gates.append(num_bits % 2)
        gates.append(temp)
        num_bits = temp

    with open(file_name2, 'w') as fh:

        if (logic_function.lower() == 'xor'):
            logic_str = '^'

        fh.write('/--***********************************************************' +
                 '***************--\n')
        fh.write('--')
        fh.write('-- Author      : Python Generated\n')
        fh.write('-- File        : ' + file_name + '\n')
        fh.write('-- Description : Implements a pipelined %s.\n' % logic_function)
        fh.write('--\n')
        fh.write('--\n')
        fh.write('--\n')
        fh.write('-- LICENSE     : SEE LICENSE FILE AGREEMENT,\n')
        fh.write('--\n')
        fh.write('--\n')
        print_header(fh)
        fh.write('--\n')
        fh.write('/--*********************************************************' +
                 '*****************--\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('module %s\n' % module_name)
        fh.write('(\n')
        fh.write('    input clk,\n')
        fh.write('    input sync_reset,\n')
        fh.write('    input valid_i,\n')
        fh.write('    input [%d downto 0) input_word,\n' % (input_width - 1))
        fh.write('    output valid_o,\n')
        fh.write('    output output_word\n')

        fh.write(');\n')
        fh.write('\n')

        for (ii, regs) in enumerate(gates):
            for idx in range(regs):
                fh.write('signal %s_%d_%d;\n' % (logic_function, ii, idx))

        fh.write('\n')
        for idx in range(num_stages):
            fh.write('signal valid%d;\n' % idx)

        fh.write('\n')
        fh.write('assign valid_o = valid%d;\n' % (num_stages - 1))
        lidx = num_stages - 1
        fh.write('assign output_word = %s_%d_%d;\n' % (logic_function, lidx, 0))
        fh.write('\n')
        fh.write('always @(posedge clk, posedge sync_reset) then\n')
        fh.write('    if (sync_reset = \'1\') then\n')
        for idx in range(num_stages):
            fh.write('        valid%d <= \'0\';\n' % idx)
        fh.write('    else\n')
        for idx in range(num_stages):
            if (idx == 0):
                fh.write('        valid0 <= valid_i;\n')
            else:
                fh.write('        valid{} <= valid{};\n'.format(idx, idx - 1))
        fh.write('    end if;\n')
        fh.write('end if;\n')
        fh.write('\n')

        fh.write('\n')
        fh.write('always @(posedge clk) then\n')
        for (ii, regs, remain_gates) in zip(count(), gates, rem_gates):
            for idx in range(regs):
                lidx = idx * 2
                ridx = lidx + 1
                insert_reg = False
                if (idx == regs - 1):
                    if (remain_gates != 0):
                        insert_reg = True
                if ii == 0:
                    if (insert_reg):
                        fh.write('    %s_%d_%d <= input_word[%d];\n'
                                 % (logic_function, ii, idx, lidx))
                    else:
                        fh.write('    %s_%d_%d <= input_word[%d] %s '
                                 'input_word[%d];\n'
                                 % (logic_function, ii, idx,
                                    lidx, logic_str, ridx))
                else:
                    if (insert_reg):
                        fh.write('    {}_{}_{} <= {}_{}_{};\n'.format(logic_function, ii, idx, logic_function,
                                 ii - 1, lidx))
                    else:
                        fh.write('    {}_{}_{} <= {}_{}_{} {} {}_{}_{};\n'.format(logic_function, ii,
                                                                                  idx, logic_function,
                                                                                  ii - 1, lidx, logic_str,
                                                                                  logic_function, ii - 1, ridx))
            fh.write('\n')
        fh.write('end if;\n')
        fh.write('\n')
        fh.write('endmodule\n')
        fh.close()


def gen_one_hot(input_width, file_path=None):
    """
        Function generates Verilog module of a one hot encoder -- simply
        explicitly implements the case statementfully pipelined mux.
        The inputs should be a single concatenated signal.
    """
    output_width = 2 ** input_width

    file_name = 'one_hot_%d_%d.v' % (input_width, output_width)

    if file_path is None:
        file_path = os.getcwd()

    file_name = os.path.join(file_path, file_name)
    hex_chars = output_width - 4

    out_msb = output_width - 1

    with open(file_name, 'w') as fh:
        fh.write('/--****************************************************--\n')
        fh.write('--')
        fh.write('-- File        : %s\n' % file_name)
        fh.write('-- Description : Implements a one_hot decoder\n')
        fh.write('-- This module has a delay of 1 clock cycles')
        fh.write('--\n')
        fh.write('-- -----------------------------------------------------\n')
        print_header(fh)
        fh.write('--\n')
        fh.write('/--***************************************************--\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('module one_hot_%d_%d\n' % (input_width, output_width))
        fh.write('(\n')
        fh.write('    input clk,\n')
        fh.write('    input [%d:0] input_word,\n' % (input_width - 1))
        fh.write('    output [%d:0] output_word\n' % out_msb)
        fh.write(');\n')

        fh.write('signal [%d:0] output_reg, next_output_reg;\n' % out_msb)
        fh.write('assign output_word = output_reg;\n')
        fh.write('\n')
        fh.write('always @(posedge clk) then\n')
        fh.write('    output_reg <= next_output_reg;\n')
        fh.write('end if;\n')
        fh.write('\n')
        fh.write('always @*\n')
        fh.write('begin\n')
        fh.write('    case (input_word)\n')
        for ii in range(output_width):
            case_str = '{0:b}'.format(ii)
            case_str = str.zfill(case_str, input_width)
            case_str = '%d''b%s' % (input_width, case_str)
            # right_str = '{0:b}'.format(2**ii)
            # right_str = str.zfill(right_str, output_width)
            hex_val = hex(2**ii)[2:]
            if hex_val[-1] == 'L':
                hex_val = hex_val[:-1]
            hex_val = str.zfill(hex_val, hex_chars)
            right_str = '%d''h%s' % (output_width, hex_val)

            fh.write('        %s : next_output_reg = %s;\n' % (case_str,
                                                               right_str))
        fh.write('    end if;\n')
        fh.write('end if;\n')
        fh.write('\n')
        fh.write('endmodule\n')


def gen_pipe_mux(input_width, output_width, file_path=None, mux_bits=2, one_hot=False, one_hot_out=False):
    """
        Function generates Verilog module of a fully pipelined mux.
        Input is a single concatenated signal.

        ==========
        Parameters
        ==========

            * input_width (int)
                input vector width
            * output_width (int)
                output vector width
            * file_path (str)
                Default current working directory.
            * mux_bits
                2**mux_bits input signals to each mux.  Input signal width
                equal to final output_width

        =======
        Returns
        =======

            Verilog file that implements a fully pipelined multiplexer.

    """

    file_name = 'pipe_mux_{}_{}.vhd'.format(input_width, output_width)
    module_name = ret_module_name(file_name)
    if file_path is None:
        file_path = os.getcwd()

    file_name2 = os.path.join(file_path, file_name)

    io_ratio = np.float(input_width / output_width)

    assert (io_ratio.is_integer()), ("Input to Output ratio must be an integer value")

    one_hot_width = int(np.ceil(input_width / float(output_width)))
    num_sels = np.ceil(np.log2(input_width / output_width)).astype(np.int)
    sels_msb = num_sels - 1

    mux_div_factor = 2**mux_bits
    num_mux_stages = np.ceil(np.float(num_sels) / mux_bits).astype(np.int)

    # TODO : calculate total delay.
    tot_delay = num_mux_stages + 1  # +1 for input delay.

    num_mux_per_stage = []
    in_width = io_ratio
    for ii in range(num_mux_stages):
        in_width = np.ceil(np.float(in_width) / mux_div_factor).astype(np.int)
        num_mux_per_stage.append(in_width)

    sel_strs = []
    reg_sel_strs = []
    if mux_bits == 1:
        for ii in range(num_mux_stages):
            temp = 'sel_d{}'.format(ii)
            sel_strs.append(temp)
    else:
        for ii in range(num_mux_stages):
            idx = (ii + 1) * mux_bits
            temp_str = []
            temp2_str = []
            for jj in range(mux_div_factor - 1):
                if (ii != num_mux_stages - 1):
                    for nn in range(num_mux_per_stage[ii]):
                        tup_val = (ii, nn, jj, idx - 1, idx - 2)
                        tup_val2 = (ii, nn, jj)
                        temp = 'sel_d{}_{}_{}({} downto {})'.format(*tup_val)
                        temp2 = 'sel_d{}_{}_{}'.format(*tup_val2)
                        temp_str.append(temp)
                        temp2_str.append(temp2)
                else:
                    if (idx > num_sels):
                        temp = 'sel_d{}_{}({})'.format(ii, jj, idx - 2)
                        temp2 = 'sel_d{}_{}'.format(ii, jj)
                    else:
                        temp = 'sel_d{}_{}({} downto {})'.format(ii, jj, idx - 1, idx - 2)
                        temp2 = 'sel_d{}_{}'.format(ii, jj)
                    temp_str.append(temp)
                    temp2_str.append(temp2)
            sel_strs.append(temp_str)
            reg_sel_strs.append(temp2_str)

    with open(file_name2, 'w') as fh:

        fh.write('----------------------------------------------------------------------\n')
        fh.write('--')
        fh.write('-- File        : {}\n'.format(file_name))
        fh.write('-- Description : Implements a pipelined multiplexer to be used in high speed design\n')
        fh.write('-- This module has a delay of {} clock cycles'.format(tot_delay))
        fh.write('--\n')
        fh.write('---------------------------------------------------------------------\n')
        # print_header(fh)
        fh.write('\n')
        fh.write('\n')
        print_intro(fh, module_name)
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('            valid_i : in std_logic;\n')
        if one_hot:
            fh.write('            sel : in std_logic_vector({} downto 0);\n'.format(one_hot_width - 1))
        else:
            fh.write('            sel : in std_logic_vector({} downto 0);\n'.format(num_sels - 1))
        fh.write('            input_word : in std_logic_vector({} downto 0);\n'.format(input_width - 1))
        fh.write('            valid_o : out std_logic;\n')
        if one_hot_out:
            fh.write('            sel_o : out std_logic_vector({} downto 0);\n'.format(one_hot_width - 1))
        else:
            fh.write('            sel_o : out std_logic_vector({} downto 0);\n'.format(num_sels - 1))
        if output_width == 1:
            fh.write('            output_word : out std_logic;\n')
        else:
            fh.write('            output_word : out std_logic_vector({} downto 0)\n'.format(output_width - 1))
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    port\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('        valid_i : in std_logic;\n')
        if one_hot:
            fh.write('        sel : in std_logic_vector({} downto 0);\n'.format(one_hot_width - 1))
        else:
            fh.write('        sel : in std_logic_vector({} downto 0);\n'.format(num_sels - 1))
        fh.write('        input_word : in std_logic_vector({} downto 0);\n'.format(input_width - 1))
        fh.write('        valid_o : out std_logic;\n')
        if one_hot_out:
            fh.write('        sel_o : out std_logic_vector({} downto 0);\n'.format(one_hot_width - 1))
        else:
            fh.write('        sel_o : out std_logic_vector({} downto 0);\n'.format(num_sels - 1))
        if output_width == 1:
            fh.write('        output_word : out std_logic;\n')
        else:
            fh.write('        output_word : out std_logic_vector({} downto 0)\n'.format(output_width - 1))
        fh.write('    );\n')
        fh.write('end {};'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n'.format(module_name))
        fh.write('\n')
        fh.write('-- logic uses 4 to 1 decoders -- generally efficient with modern FPGAs\n')
        fh.write('attribute keep : string;\n')
        fh.write('\n')
        sel_sens_list = ''
        for index in range(num_mux_stages):
            for str_val in reg_sel_strs[index]:
                fh.write('signal {} : std_logic_vector({} downto 0);\n'.format(str_val, sels_msb))
                fh.write('attribute keep of {} : signal is "true";\n'.format(str_val))
                fh.write('\n')
                sel_sens_list += str_val + ', '
        if ~one_hot_out:
            fh.write('signal sel_d_out : std_logic_vector({} downto 0);\n'.format(sels_msb))
        for ii in range(num_mux_stages + 1):
            fh.write('signal valid_d{} : std_logic;\n'.format(ii))

        mux_sens_list = ''
        if output_width == 1:
            for (ii, depth) in enumerate(num_mux_per_stage):
                if depth == 1:
                    fh.write('signal mux{}, next_mux{} : std_logic;\n'.format(ii, ii))
                else:
                    fh.write('signal mux{}, next_mux{} : std_logic_vector({} downto 0);\n'.format(ii, ii, depth - 1))
                mux_sens_list += 'mux{}'.format(ii) + ', '
        else:
            for (ii, depth) in enumerate(num_mux_per_stage):
                if depth == 1:
                    str_val = 'mux{}'.format(ii)
                    fh.write('signal {}, next_{} : std_logic_vector({} downto 0);\n'.format(str_val, str_val, output_width - 1))
                    mux_sens_list += str_val + ', '
                else:
                    for jj in range(depth):
                        str_val = 'mux{}_{}'.format(ii, jj)
                        fh.write('signal {}, next_{} : std_logic_vector({} downto 0);\n'.format(str_val, str_val, output_width - 1))
                        mux_sens_list += str_val + ', '

        fh.write('signal input_word_d : std_logic_vector({} downto 0);\n'.format(input_width - 1))
        if one_hot:
            fh.write('signal next_sel : std_logic_vector({} downto 0);\n'.format(num_sels - 1))
            if one_hot_out:
                for ii in range(num_mux_stages + 1):
                    fh.write('signal one_hot_d{} : std_logic_vector({} downto 0);\n'.format(ii, one_hot_width - 1))
        fh.write('\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('output_word <= mux{};\n'.format(num_mux_stages - 1))
        fh.write('valid_o <= valid_d{};\n'.format(num_mux_stages))
        if one_hot_out:
            fh.write('sel_o <= one_hot_d{};\n'.format(num_mux_stages))
        else:
            fh.write('sel_o <= sel_d_out;\n')
        fh.write('\n')
        # fh.write('integer ii;\n')
        fh.write('\n')
        fh.write('main_clk_proc: process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        for ii in range(num_mux_stages + 1):
            fh.write('            valid_d{}  <= \'0\';\n'.format(ii))

        fh.write('        else\n')
        fh.write('            valid_d0  <= valid_i;\n')
        for ii in range(num_mux_stages):
            fh.write('            valid_d{}  <= valid_d{};\n'.format(ii + 1, ii))

        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process main_clk_proc;\n')
        fh.write('\n')
        fh.write('delay_proc : process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        input_word_d <= input_word;\n')
        for (ii, depth) in enumerate(num_mux_per_stage):
            if depth == 1:
                fh.write('        mux{} <= next_mux{};\n'.format(ii, ii))
            else:
                for jj in range(depth):
                    fh.write('        mux{}_{} <= next_mux{}_{};\n'.format(ii, jj, ii, jj))
        if one_hot:
            for str_val in reg_sel_strs[0]:
                    fh.write('        {} <= next_sel;\n'.format(str_val))
        else:
            for str_val in reg_sel_strs[0]:
                fh.write('        {} <= sel;\n'.format(str_val))

        for ii in range(1, num_mux_stages):
            for (jj, str_val) in enumerate(reg_sel_strs[ii]):
                offset = num_mux_per_stage[ii]
                str_val_rhs = reg_sel_strs[ii - 1][jj]
                fh.write('        {} <= {};\n'.format(str_val, str_val_rhs))
        if one_hot_out:
            fh.write('        one_hot_d0 <= sel;\n')
            for ii in range(num_mux_stages):
                fh.write('        one_hot_d{} <= one_hot_d{};\n'.format(ii + 1, ii))
        if ~one_hot_out:
            fh.write('        sel_d_out <= {};\n'.format(reg_sel_strs[-1][-1]))
        fh.write('    end if;\n')
        fh.write('end process delay_proc;\n')
        fh.write('\n')
        if one_hot:
            fh.write('one_hot : process(sel)\n')
            fh.write('begin\n')
            for jj in range(num_sels):
                fh.write('    next_sel({}) <= '.format(jj))
                or_prefix = ''
                for index in range(1, one_hot_width + 1):
                    sel_val = index - 1
                    bit_val = fp_utils.dec_to_ubin(sel_val, num_bits=num_sels)[-1 - jj]

                    if (bit_val == '1'):
                        fh.write('{}sel({})'.format(or_prefix, sel_val))
                        or_prefix = ' or '

                fh.write(';\n')
            fh.write('end process one_hot;\n')
        fh.write('\n')
        sens_list = sel_sens_list + mux_sens_list + 'input_word_d'
        fh.write('mux_proc : process({})\n'.format(sens_list))
        fh.write('begin\n')
        for (ii, depth) in enumerate(num_mux_per_stage):
            if depth == 1:
                fh.write('    next_mux{} <= mux{};\n'.format(ii, ii))
            else:
                for jj in range(depth):
                    fh.write('    next_mux{}_{} <= mux{}_{};\n'.format(ii, jj, ii, jj))
        fh.write('\n')

        incr = 2**mux_bits
        num_sels_rem = num_sels
        for ii in range(num_mux_stages):
            if ii == 0:
                rhs = 'input_word_d('
            else:
                rhs = 'mux{}_'.format(ii - 1)
            max_idx = input_width - 1
            if (ii > 0):
                max_idx = num_mux_per_stage[ii - 1] - 1

            incr = 2**(np.min((num_sels_rem, mux_bits)))
            if (ii != num_mux_stages - 1):
                for nn in range(num_mux_per_stage[ii]):
                    for jj in range(incr):
                        offset = nn + (jj * num_mux_per_stage[ii])
                        if (jj != incr - 1):
                            str_val_lhs = sel_strs[ii][offset]
                        if (jj == 0):
                            fh.write('    if ({} = 0) then\n'.format(str_val_lhs))
                        else:
                            if (jj == incr - 1):
                                fh.write('    else\n')
                            else:
                                fh.write('    elsif ({} = {}) then\n'.format(str_val_lhs, jj))
                        if ii == 0:
                            idx0 = (nn * incr + jj) * output_width
                            if output_width > 1:
                                idx1 = idx0 + output_width - 1
                                if (idx1 > max_idx):
                                    idx1 = max_idx
                                    idx0 = idx1 - output_width + 1
                                rhs_str = '{} downto {})'.format(idx1, idx0)
                            else:
                                rhs_str = '{})'.format(idx0)
                        else:
                            idx = nn * incr + jj
                            if (idx > max_idx):
                                idx = max_idx
                            rhs_str = '{}'.format(idx)
                        if (num_mux_per_stage[ii] > 1):
                            fh.write('        next_mux{}_{} <= '.format(ii, nn) + rhs + rhs_str + ';\n')
                        else:
                            fh.write('        next_mux{} <= '.format(ii) + rhs + rhs_str + ';\n')
                    fh.write('    end if;\n\n')
            else:
                for jj in range(incr):
                    bin_val = '{0:08b}'.format(jj)
                    bin_val = bin_val[-num_sels:]
                    if (jj == 0):
                        fh.write('    if ({} = \"{}\") then\n'.format(sel_strs[ii][jj], bin_val))
                    else:
                        if (jj == incr - 1):
                            fh.write('    else\n')
                        else:
                            fh.write('    elsif ({} = {}) then\n'.format(sel_strs[ii][jj], bin_val))
                    for nn in range(num_mux_per_stage[ii]):
                        if ((ii == 0) & (output_width > 1)):
                            idx0 = (nn * incr + jj) * output_width
                            idx1 = idx0 + output_width - 1
                            if (idx1 > max_idx):
                                idx1 = max_idx
                                idx0 = idx1 - output_width + 1
                            rhs_str = '{} downto {})'.format(idx1, idx0)
                        else:
                            idx = nn * incr + jj
                            if idx > max_idx:
                                idx = max_idx
                            rhs_str = '{}'.format(idx)

                        if (num_mux_per_stage[ii] > 1):
                            fh.write('        next_mux{}_{} <= '.format((ii, nn)) + rhs + rhs_str + ';\n')
                        else:
                            fh.write('        next_mux{} <= '.format(ii) + rhs + rhs_str + ';\n')
                fh.write('    end if;\n')
            num_sels_rem -= mux_bits

        fh.write('end process mux_proc;\n')
        fh.write('\n')
        fh.write('end rtl;\n')
    return file_name


def gen_slicer(file_path=None, input_width=48, output_width=16, input_base=None, max_offset=31, rev_dir=False):
    """
        Function generates Verilog module of for a configurable slicer.
        The output is a sliced version of the input.  The offset port defines
        the LSB offset from the bottom of the input bit stack.

        ==========
        Parameters
        ==========

            * input_width (int)
                input vector width
            * output_width (int)
                output vector width
            * file_path (str)
                Default current working directory.
            * max_offset
                Maximum user defined bit offset.
            * input_base
                Defines the true base of the input vector.  Allows the
                slicer to slice the actual occupied bits.
            * rev_dir
                Boolean to indicate whether slice offset begins counting
                from the LSB or offset from the MSB.
                Default : relative to LSB.


        =======
        Returns
        =======

            Verilog file that implements a slicer module.

    """
    module_name = 'slicer{}_{}'.format(input_width, output_width)
    file_name = '{}.vhd'.format(module_name)
    out_msb = output_width - 1
    ctrl_bits = fp_utils.ret_num_bitsU(max_offset)

    if input_base is None:
        diff_bits = 0
    else:
        diff_bits = input_base - output_width

    if file_path is None:
        file_path = os.getcwd()

    file_name2 = os.path.join(file_path, file_name)


    with open(file_name2, 'w') as fh:
        fh.write('\n')
        fh.write('--*******************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Python Generated\n')
        fh.write('-- File        : {}.\n'.format(module_name))
        fh.write('-- Description : Generates a variable slicer module.\n')
        fh.write('--\n')
        # print_header(fh)
        fh.write('--\n')
        fh.write('--*********************************************************' +
                 '*****************--\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')

        fh.write('package {}_cmp is\n'.format(module_name))
        fh.write('    component {}\n'.format(module_name))
        fh.write('        port \n')
        fh.write('        ( \n')
        fh.write('           sync_reset : in std_logic;\n')
        fh.write('           clk : in std_logic;\n')
        fh.write('\n')
        fh.write('           -- Settings offet the slicer from the base value.\n')
        fh.write('           slice_offset_i : in std_logic_vector({} downto 0);\n'.format(ctrl_bits - 1)) # analysis:ignore
        fh.write('\n')
        fh.write('           valid_i : in std_logic;  -- Data Valid Signal.\n')
        fh.write('           signal_i : in std_logic_vector({} downto 0);\n'.format(input_width - 1))
        fh.write('\n')
        fh.write('           valid_o : out std_logic;\n')
        fh.write('           signal_o : out std_logic_vector({} downto 0)\n'.format(output_width - 1))
        fh.write('        );\n')
        fh.write('    end component;\n')
        fh.write('end package {}_cmp;\n'.format(module_name))
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    port \n')
        fh.write('    ( \n')
        fh.write('       sync_reset : in std_logic;\n')
        fh.write('       clk : in std_logic;\n')
        fh.write('\n')
        fh.write('       -- Settings offet the slicer from the base value.\n')
        fh.write('       slice_offset_i : in std_logic_vector({} downto 0);\n'.format(ctrl_bits - 1)) # analysis:ignore
        fh.write('\n')
        fh.write('       valid_i : in std_logic;  -- Data Valid Signal.\n')
        fh.write('       signal_i : in std_logic_vector({} downto 0);\n'.format(input_width - 1))
        fh.write('\n')
        fh.write('       valid_o : out std_logic;\n')
        fh.write('       signal_o : out std_logic_vector({} downto 0)\n'.format(output_width - 1))
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is\n'.format(module_name))
        fh.write('\tsignal valid_d : std_logic;\n')
        fh.write('\n')
        fh.write('\tsignal output_reg, next_output_reg : std_logic_vector({} downto 0);\n'.format(out_msb))
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('signal_o <= output_reg;\n')
        fh.write('valid_o <= valid_d;\n')
        fh.write('\n')
        fh.write('sync_proc:\n')
        fh.write('process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        fh.write('            output_reg <= (others => \'0\');\n')
        fh.write('            valid_d <= \'0\';\n')
        fh.write('        else\n')
        fh.write('            output_reg <= next_output_reg;\n')
        fh.write('            valid_d <= valid_i;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('async_proc:\n')
        fh.write('process(slice_offset_i, signal_i)\n')
        fh.write('begin\n')
        fh.write('    switch : case slice_offset_i is\n')
        for ii in range(2**ctrl_bits):
            bit_str = '{0:b}'.format(ii)
            bit_str = str.zfill(bit_str, ctrl_bits)
            if rev_dir is True:
                if ii <= max_offset:
                    lhs = input_width - 1 - ii - diff_bits
                    rhs = lhs - output_width + 1
                else:
                    lhs = input_width - 1 - max_offset
                    rhs = lhs - output_width + 1
            else:
                if ii <= max_offset:
                    lhs = ii + diff_bits + output_width - 1
                    rhs = ii + diff_bits
                else:
                    lhs = max_offset + diff_bits + output_width - 1
                    rhs = max_offset + diff_bits
            fh.write('        when \"{}\" => next_output_reg <= signal_i({} downto {});\n'.format(bit_str, lhs, rhs))  #analysis:ignore
        fh.write('        when others => next_output_reg <= signal_i({} downto {});\n'.format(lhs, rhs))
        fh.write('\n')
        fh.write('    end case switch;\n')
        fh.write('\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('end rtl;\n')
        fh.close()

        return file_name, module_name


def gen_exp_conv(file_name, project_file, combined_table, corr_fac_fi, input_width=12, shift_bits=4, table_bits=16,
                 device='xc6slx45t', family='spartan6', package='csg324', speed_grade=3):

    assert(file_name is not None), 'User must specify File Name'
    module_name = ret_module_name(file_name)   # file_name[idx:idx + idx2]
    word_length = corr_fac_fi.word_length
    # check number of 1's in binary
    num_ones = 0
    for bit in corr_fac_fi.bin[2:]:
        num_ones += int(bit)

    shift = False
    block_latency = 12
    delay_to_mux = 8
    if (num_ones == 1):
        # constant multiplier reduces to a shift
        block_latency = 10
        shift = True
        shift_value = int(np.ceil(np.abs(np.log2(corr_fac_fi.double))))
        word_length = shift_value
        corr_fac_fi = fp_utils.ufi(0, word_length, word_length)
        delay_to_mux = 6

    corr_mult_bits = word_length + (input_width - 2)

    frac_fi = fp_utils.ufi(0, input_width - 2, input_width - 2)
    b_port_fi = fp_utils.mult_fi(frac_fi, corr_fac_fi)
    table_fi = fp_utils.ufi(0, table_bits, table_bits - 1)

    c_port_fi = fp_utils.mult_fi(b_port_fi, table_fi)

    dsp_slice_msb = c_port_fi.fraction_length
    interp_width = table_bits + corr_mult_bits

    offset = file_name.rfind('.')
    exp_name = ret_file_name(file_name[:offset])
    proj_dir = os.path.dirname(os.path.realpath(project_file)) + '/'

    correction_name = exp_name + '_CorrDSP'
    correction_file = proj_dir + correction_name + '.xco'

    corr_fac_name = exp_name + '_CorrMult'
    corr_fac_file = proj_dir + corr_fac_name + '.xco'

    exp_rom_name = exp_name + '_ROM'
    exp_rom_file = proj_dir + exp_rom_name + '.xco'

    exp_rom_name = exp_name + '_large_table'
    exp_rom_coe = proj_dir + exp_rom_name + '.coe'
    exp_rom_file = proj_dir + exp_rom_name + '.xco'

    shift_msb = shift_bits - 1

    with open(file_name, 'w') as fh:
        fh.write('\n')
        fh.write('--************************************************************'
                 '**************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : %s.v\n' % module_name)
        fh.write('-- Description : Module converts an exponential value to '
                 'linear.\n')
        fh.write('--               The module uses a correction multiplier to '
                 'improve accuracy\n')
        fh.write('--\n')
        fh.write('--\n')
        print_header(fh)
        fh.write('\n')
        fh.write('--\n')
        fh.write('--************************************************************'
                 '**************--\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('module %s\n' % module_name)
        fh.write('#(parameter INPUT_WIDTH = %d,\n' % input_width)
        fh.write('  parameter TABLE_BITS=%d)\n' % table_bits)
        fh.write('(\n')
        fh.write('  input sync_reset,\n')
        fh.write('  input clk,\n')
        fh.write('\n')
        fh.write('  input valid_i,\n')
        fh.write('  input [INPUT_WIDTH-1:0] log_i,\n')
        fh.write('\n')
        fh.write('  output valid_o,\n')
        fh.write('  output [TABLE_BITS-1:0] lin_val_o,\n')
        fh.write('  output [%d:0] shift_val_o\n' % shift_msb)
        fh.write('\n')
        fh.write(');\n')
        fh.write('\n')
        sl_width = input_width // 2
        sl_msb = sl_width - 1

        fh.write('parameter UPPER_SL_WIDTH = INPUT_WIDTH/2;\n')
        fh.write('parameter LOWER_SL_WIDTH = INPUT_WIDTH/2;\n')
        fh.write('parameter SLICE_MSB = UPPER_SL_WIDTH + LOWER_SL_WIDTH - 1;\n')
        fh.write('parameter BLOCK_LATENCY = {};\n'.format(block_latency))
        fh.write('parameter SHIFT_MSB = TABLE_BITS + %d;\n' % shift_msb)
        fh.write('\n')
        interp_bits = (table_bits + corr_mult_bits)
        fh.write('parameter INTERP_WIDTH = {};\n'.format(interp_bits))
        fh.write('\n')
        str1 = (input_width // 2) * '1'
        str0 = (input_width // 2) * '0'
        fh.write('parameter ALL_ONES  = {}\'b{};\n'.format(input_width // 2, str1))
        fh.write('parameter ALL_ZEROS = {}\'b{};\n'.format(input_width // 2, str0))
        fh.write('\n')
        fh.write('signal upper_slice : std_logic_vector({} downto 0);\n'.format(sl_msb))
        fh.write('signal lower_slice : std_logic_vector({} downto 0);\n'.format(sl_msb))
        fh.write('\n')
        fh.write('signal lower_slice_d0, lower_slice_d1, lower_slice_d2 : std_logic_vector({} downto 0);\n'.format(sl_msb))

        for i in range(block_latency - 1):
            fh.write('signal upper_slice_d{} : std_logic_vector({} downto 0);\n'.format(i, sl_msb))
        fh.write('\n')
        for i in range(delay_to_mux):
            fh.write('signal lower_table_d{} : std_logic_vector(TABLE_BITS-1 downto 0);\n')

        fh.write('signal upper_table_d, upper_table_d2 : std_logic_vector(TABLE_BITS-1 downto 0);\n')
        fh.write('\n')
        fh.write('signal std_logic_vector({} downto 0) upper_shift_d [{}:0];\n'.format(shift_msb, delay_to_mux - 1))

        for i in range(delay_to_mux):
            fh.write('signal lower_shift_d{} : std_logic_vector({} downto 0);\n'.format(i, shift_msb))

        fh.write('-- small table is registered 5 times\n')
        fh.write('-- max latency of distributed mem is 2.\n')
        fh.write('\n')
        fh.write('signal upper_table_pad : std_logic_vector(INTERP_WIDTH-1 downto 0);\n')
        fh.write('\n')
        # fh.write('signal [TABLE_BITS-1:0] b_factor;\n')
        # fh.write('\n')
        fh.write('signal dsp_out : std_logic_vector(INTERP_WIDTH downto 0);\n')
        fh.write('signal dsp_out_slice : std_logic_vector(TABLE_BITS-1 downto 0);\n')
        fh.write('\n')
        fh.write('signal corr_value : std_logic_vector({} downto 0);\n'.format(corr_mult_bits - 1))
        if (shift is True):
            fh.write('signal corr_value_d1 : std_logic_vector({} downto 0);\n'.format(corr_mult_bits - 1))
            fh.write('signal corr_value_d2 : std_logic_vector({} downto 0);\n'.format(corr_mult_bits - 1))
        fh.write('\n')
        fh.write('signal mux_out, next_mux_out : std_logic_vector(TABLE_BITS-1 downto 0);\n')
        fh.write('signal shift_mux_out, next_shift_mux_out : std_logic_vector({} downto 0);\n'.format(shift_msb))
        fh.write('\n')
        fh.write('signal [BLOCK_LATENCY-1:0] valid_d;\n')
        fh.write('\n')
        fh.write('signal mux_sw, next_mux_sw;\n')
        fh.write('\n')
        fh.write('signal [TABLE_BITS+%d:0] upper_table, lower_table;\n' % shift_msb)
        fh.write('\n')
        fh.write('signal [TABLE_BITS-1:0] u_table, l_table;\n')
        fh.write('signal [%d:0] u_shift, l_shift;\n' % shift_msb)
        fh.write('\n')
        fh.write('signal [UPPER_SL_WIDTH:0] addra, addrb;\n')
        fh.write('\n')
        fh.write('assign u_table = upper_table[TABLE_BITS-1:0];\n')
        fh.write('assign l_table = lower_table[TABLE_BITS-1:0];\n')
        fh.write('\n')
        fh.write('assign u_shift = upper_table[SHIFT_MSB:TABLE_BITS];\n')
        fh.write('assign l_shift = lower_table[SHIFT_MSB:TABLE_BITS];\n')
        fh.write('\n')
        str0 = b_port_fi.word_length * '0'
        fh.write('assign upper_table_pad = '
                 '{upper_table_d2,%d\'b%s};\n' % (b_port_fi.word_length, str0))
        fh.write('\n')
        fh.write('assign dsp_out_slice = dsp_out[%d:%d];\n'
                 % (dsp_slice_msb, dsp_slice_msb - table_bits + 1))
        fh.write('\n')
        fh.write('assign valid_o = valid_d[BLOCK_LATENCY-1];\n')
        fh.write('assign lin_val_o = mux_out;\n')
        fh.write('assign shift_val_o = shift_mux_out;\n')
        fh.write('assign upper_slice = log_i[SLICE_MSB:LOWER_SL_WIDTH];\n')
        fh.write('assign lower_slice = log_i[LOWER_SL_WIDTH-1:0];\n')
        fh.write('assign addra = {\'1\',upper_slice};\n')
        fh.write('assign addrb = {\'0\',lower_slice};\n')
        str_val = str(shift_value) + '\'b' + shift_value * '0'
        if (shift is True):
            fh.write('assign corr_value = {%s, lower_slice_d[2]};\n' % str_val)
        fh.write('\n')
        fh.write('integer ii;\n')
        fh.write('\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('\n')
        if (shift is True):
            fh.write('    corr_value_d1 <= corr_value;\n')
            fh.write('    corr_value_d2 <= corr_value_d1;\n')

        fh.write('    lower_table_d[0] <= l_table;\n')
        fh.write('    for (ii = 1; ii < %d; ii = ii + 1) then\n' % delay_to_mux)
        fh.write('        lower_table_d[ii]  <= lower_table_d[ii-1];\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('    lower_slice_d[0] <= lower_slice;\n')
        fh.write('    for (ii = 1; ii < 3; ii = ii + 1) then\n')
        fh.write('        lower_slice_d[ii]  <= lower_slice_d[ii-1];\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('    upper_slice_d[0] <= upper_slice;\n')
        fh.write('    for (ii = 1; ii < BLOCK_LATENCY-1; ii = ii + 1) then\n')
        fh.write('        upper_slice_d[ii] <= upper_slice_d[ii-1];\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('    upper_shift_d[0] <= u_shift;\n')
        fh.write('    for (ii = 1; ii < %d; ii = ii + 1) then\n' % delay_to_mux)
        fh.write('        upper_shift_d[ii] <= upper_shift_d[ii-1];\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('    upper_table_d <= u_table;\n')
        fh.write('    upper_table_d2 <= upper_table_d;\n')
        fh.write('\n')
        fh.write('    lower_shift_d[0] <= l_shift;\n')
        fh.write('    for (ii = 1; ii < %d; ii = ii + 1) then\n' % delay_to_mux)
        fh.write('        lower_shift_d[ii] <= lower_shift_d[ii-1];\n')
        fh.write('    end if;\n')
        fh.write('\n')
        fh.write('end if;\n')
        fh.write('\n')
        fh.write('process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (sync_reset = \'1\') then\n')
        fh.write('        valid_d <= 0;\n')
        fh.write('        mux_out <= 0;\n')
        fh.write('        shift_mux_out <= 0;\n')
        fh.write('        mux_sw <= 0;\n')
        fh.write('    else\n')
        fh.write('        valid_d <= {valid_d[BLOCK_LATENCY-2:0],valid_i};\n')
        fh.write('        mux_out <= next_mux_out;\n')
        fh.write('        shift_mux_out <= next_shift_mux_out;\n')
        fh.write('        mux_sw <= next_mux_sw;\n')
        fh.write('    end if;\n')
        fh.write('end if;\n')
        fh.write('\n')
        fh.write('--mux Process Latency 1.\n')
        fh.write('always @*\n')
        delay_val = delay_to_mux - 1
        fh.write('begin\n')
        fh.write('    if ((upper_slice_d[BLOCK_LATENCY-2] == ALL_ONES) '
                 '|| (upper_slice_d[BLOCK_LATENCY-2] == ALL_ZEROS)) then\n')
        fh.write('        next_mux_sw = \'1\';\n')
        fh.write('        next_mux_out = lower_table_d[%d];\n' % delay_val)
        fh.write('        next_shift_mux_out = lower_shift_d[%d];\n' % delay_val)
        fh.write('    else\n')
        fh.write('        next_mux_sw = \'0\';\n')
        fh.write('        next_mux_out = dsp_out_slice;\n')
        fh.write('        next_shift_mux_out = upper_shift_d[%d];\n' % delay_val)
        fh.write('    end if;\n')
        fh.write('end if;\n')
        fh.write('\n')
        fh.write('-- Latency = 3.\n')
        fh.write('%s Table (\n' % exp_rom_name)
        fh.write('  .clka(clk),\n')
        fh.write('  .addra(addra),\n')
        fh.write('  .douta(upper_table),\n')
        fh.write('  .clkb(clk),\n')
        fh.write('  .addrb(addrb),\n')
        fh.write('  .doutb(lower_table)\n')
        fh.write(');\n')
        fh.write('\n')
        if shift is False:
            fh.write('%s CorrMult (\n' % corr_fac_name)
            fh.write('  .clk(clk),\n')
            fh.write('  .a(lower_slice_d[2]),\n')
            fh.write('  .p(corr_value)\n')
            fh.write(');\n')
        fh.write('\n')
        fh.write('--Latency = 6. AGC Corr DSP\n')
        fh.write('%s CorrDSP (\n' % correction_name)
        fh.write('  .clk(clk),\n')
        fh.write('  .a(upper_table_d2),\n')
        if shift is False:
            fh.write('  .b(corr_value),\n')
        else:
            fh.write('  .b(corr_value_d2),\n')
        fh.write('  .c(upper_table_pad),\n')
        fh.write('  .p(dsp_out)\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('endmodule\n')
        fh.close()

        if shift is False:
            x_utils.xco_mult(corr_fac_file,
                             project_file,
                             input_a_width=input_width--2,
                             input_b_width=input_width--2,
                             device=device,
                             family=family,
                             package=package,
                             speed_grade=speed_grade,
                             output_width=corr_mult_bits-1,
                             constant_value=corr_fac_fi.udec,
                             mult_type='Constant_Coefficient_Multiplier',
                             multiplier_construction='Use_LUTs',
                             pipe_stages=2)

        x_utils.coe_write(combined_table, radix=16, file_name=exp_rom_coe)

        x_utils.xco_dsp48_macro(correction_file, project_file,
                                a_width=table_bits,
                                b_width=corr_mult_bits,
                                device=device,
                                family=family,
                                package=package,
                                speed_grade=speed_grade,
                                c_width=interp_width,
                                areg_1=True,
                                areg_2=True,
                                areg_3=True,
                                areg_4=True,
                                breg_1=True,
                                breg_2=True,
                                breg_3=True,
                                breg_4=True,
                                creg_1=True,
                                creg_2=True,
                                creg_3=True,
                                creg_4=True,
                                creg_5=True,
                                sclr=False,
                                instruction1='A*B+C')

        x_utils.xco_block_rom(exp_rom_file, project_file,
                              exp_rom_coe,
                              dual_port=True,
                              device=device,
                              family=family,
                              package=package,
                              speed_grade=speed_grade,
                              data_width=shift_bits + table_bits,
                              depth=combined_table.len)


def gen_log_conv(path, combined_table, type_bits=0, tuser_width=0, tlast=False, prefix=''):

    assert(path is not None), 'User must specify Path'
    path = ret_valid_path(path)

    input_width = fp_utils.nextpow2(combined_table.len // 2) * 2
    word_length = combined_table.qvec[0]
    output_width = word_length // 2
    if tuser_width > 0:
        tuser_msb = tuser_width - 1

    id_val = 0
    if tlast:
        id_val += 1
    if tuser_width:
        id_val += 2


    file_name = path + '{}axi_log_conv_{}iw_{}ow_{}.vhd'.format(prefix, input_width, output_width, id_val)

    module_name = ret_module_name(file_name)
    word_msb = word_length - 1
    # there is a delay of 8 . 3 for ROM, 1 for Mux, 4 for dsp.  # use a 16 deep fifo
    addr_width = 4
    af_thresh = 7

    input_msb = input_width - 1
    output_msb = output_width - 1
    addr_bits = input_width // 2

    table_msb = word_length - 1
    interp_width = output_width + addr_bits
    type_msb = type_bits - 1

    # offset = file_name.rfind('.')
    # top_name = ret_file_name(file_name[:offset])
    funcs = '(A * B) + C'
    (dsp_file, dsp_name) = gen_dsp48E1(path, module_name, opcode=funcs, use_ce=False, areg=2, breg=2,
                                       b_signed=False, a_width=output_width, b_width=addr_bits+1, c_width=interp_width+1)
    print(dsp_name)

    # axis buffer
    (_, fifo_name) = gen_axi_fifo(path, tuser_width=tuser_width, tlast=tlast, almost_full=True, ram_style='distributed')
    print(fifo_name)

    # combined table
    (table_file, table_name) = gen_rom(path, combined_table, rom_type='dp', rom_style='block', prefix='{}_'.format(module_name))

    print(table_name)

    sl_width = input_width // 2
    sl_msb = sl_width - 1

    with open(file_name, 'w') as fh:
        fh.write('\n')
        fh.write('--****************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : %s.vhd\n' % module_name)
        fh.write('-- Description : Module converts an input Magnitude to a log value.  Note that\n')
        fh.write('--               this is a Natural log conversion (better compression of the\n')
        fh.write('--               signal.  The module uses linear interpolationto reduce table\n')
        fh.write('--               size and improve accuracy.\n')
        fh.write('--\n')
        fh.write('\n')
        print_header(fh)
        fh.write('--\n')
        fh.write('--****************************************************************************--\n')
        fh.write('\n')
        fh.write('\n')
        print_libraries(fh)

        fh.write('package %s_cmp is\n' % module_name)
        fh.write('    component %s\n' % module_name)
        if tuser_width:
            fh.write('        generic\n')
            fh.write('        (\n')
            fh.write('            TUSER_WIDTH : integer := 8\n')
            fh.write('        );\n')
        fh.write('        port \n')
        fh.write('        ( \n')
        fh.write('            clk : in std_logic;\n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('            s_axis_tvalid : in std_logic;\n')
        fh.write('            s_axis_tdata : in std_logic_vector({} downto 0);\n'.format(input_msb))
        if tuser_width > 0:
            fh.write('            s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n'.format(tuser_msb))

        if tlast:
            fh.write('            s_axis_tlast : in std_logic;\n')
        fh.write('            s_axis_tready : out std_logic;\n')
        fh.write('\n')

        fh.write('            m_axis_tvalid : out std_logic;\n')
        fh.write('            m_axis_tdata : out std_logic_vector({} downto 0);\n'.format(output_msb)) # analysis:ignore
        if tuser_width > 0:
            fh.write('            m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('            m_axis_tlast : out std_logic;\n')

        fh.write('            m_axis_tready : in std_logic\n')
        fh.write('        );\n')
        fh.write('    end component;\n')
        fh.write('end package {}_cmp;\n'.format(module_name))
        fh.write('\n')
        print_libraries(fh)
        fh.write('library work;\n')
        fh.write('use work.{}_cmp.all;\n'.format(fifo_name))
        fh.write('use work.{}_cmp.all;\n'.format(dsp_name))
        fh.write('use work.{}_cmp.all;\n'.format(table_name))
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        if tuser_width:
            fh.write('    generic\n')
            fh.write('    (\n')
            fh.write('            TUSER_WIDTH : integer := 8\n')
            fh.write('    );\n')
        fh.write('    port \n')
        fh.write('    ( \n')
        fh.write('        clk : in std_logic;\n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('        s_axis_tvalid : in std_logic;\n')
        fh.write('        s_axis_tdata : in std_logic_vector({} downto 0);\n'.format(input_msb))
        if tuser_width > 0:
            fh.write('        s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('        s_axis_tlast : in std_logic;\n')
        fh.write('        s_axis_tready : out std_logic;\n')
        fh.write('\n')
        fh.write('        m_axis_tvalid : out std_logic;\n')
        fh.write('        m_axis_tdata : out std_logic_vector({} downto 0);\n'.format(output_msb))
        if tuser_width > 0:
            fh.write('        m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('        m_axis_tlast : out std_logic;\n')

        fh.write('        m_axis_tready : in std_logic\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is\n\n'.format(module_name))
        one_str = addr_bits * '1'
        zero_str = addr_bits * '0'
        fh.write('\tconstant ALL_ONES : std_logic_vector({} downto 0) := "{}";\n'.format(addr_bits-1, one_str))
        fh.write('\tconstant ALL_ZEROS : std_logic_vector({} downto 0) := "{}";\n'.format(addr_bits-1, zero_str))
        fh.write('\n')
        fh.write('\tsignal upper_slice : std_logic_vector({} downto 0);\n'.format(sl_msb))
        fh.write('\tsignal lower_slice : std_logic_vector({} downto 0);\n'.format(sl_msb))
        fh.write('\n')
        fh.write('\ttype LOWER_ARRAY_TYPE is array (0 to 2) of std_logic_vector({} downto 0);\n'.format(sl_msb))
        fh.write('\tsignal lower_slice_d : LOWER_ARRAY_TYPE;\n')
        fh.write('\ttype UPPER_ARRAY_TYPE is array (0 to 6) of std_logic_vector({} downto 0);\n'.format(sl_msb))
        fh.write('\tsignal upper_slice_d : UPPER_ARRAY_TYPE;\n')
        fh.write('\tsignal addra : std_logic_vector({} downto 0);\n'.format(addr_bits))
        fh.write('\tsignal addrb : std_logic_vector({} downto 0);\n'.format(addr_bits))
        fh.write('\n')
        # if type_bits > 0:
        #     fh.write('\tsignal id_s0, id_s1, id_s2, id_s3, id_s4, id_s5, id_s6,'
        #              ' id_s7 : std_logic_vector({} downto 0);\n'.format(type_msb))

        if tlast:
            fh.write('\tsignal tlast_d : std_logic_vector(7 downto 0);\n')


        if tuser_width > 0:
            fh.write('\tsignal tuser_d0, tuser_d1, tuser_d2, tuser_d3, tuser_d4, tuser_d5,'
                     ' tuser_d6, tuser_d7 : std_logic_vector(TUSER_WIDTH-1 downto 0);\n')

        fh.write('\ttype SMALL_ARRAY_TYPE is array (0 to 3) of std_logic_vector({} downto 0);\n'.format(output_msb))
        fh.write('\tsignal small_table_out_d : SMALL_ARRAY_TYPE;\n')
        fh.write('\n')
        fh.write('\tsignal upper_table, lower_table : std_logic_vector({} downto 0);\n'.format(table_msb))
        fh.write('\tsignal interp_out : std_logic_vector(47 downto 0);\n')
        fh.write('\tsignal interp_out_slice : std_logic_vector({} downto 0);\n'.format(output_msb))
        fh.write('\tsignal mag_table, diff_table, mag_small_table : std_logic_vector({} downto 0);\n'.format(output_msb)) #analysis:ignore
        fh.write('\tsignal large_table_pad : std_logic_vector({} downto 0);\n'.format(interp_width))
        fh.write('\tsignal bterm : std_logic_vector({} downto 0);\n'.format(addr_bits))
        fh.write('\tsignal almost_full : std_logic;\n')
        fh.write('\n')
        fh.write('\tsignal mux_out, next_mux_out : std_logic_vector({} downto 0);\n'.format(output_msb))
        fh.write('\n')
        fh.write('\tsignal tvalid_d : std_logic_vector(7 downto 0);\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('mag_table <= upper_table({} downto {});\n'.format(table_msb, output_width))
        fh.write('diff_table <= upper_table({} downto 0);\n'.format(output_msb))
        fh.write('s_axis_tready <= \'1\' when (almost_full = \'0\') else \'0\';\n')
        fh.write('mag_small_table <= lower_table({} downto {});\n'.format(table_msb, output_width))
        fh.write('upper_slice <= s_axis_tdata({}downto {});\n'.format(input_msb, addr_bits))
        fh.write('lower_slice <= s_axis_tdata({} downto 0);\n'.format(addr_bits - 1))
        fh.write('addra <= \'1\' & upper_slice;\n')
        fh.write('addrb <= \'0\' & lower_slice;\n')
        fh.write('bterm <= \'0\' & lower_slice_d(2);\n')
        tuple_val = (output_msb, zero_str)
        fh.write('large_table_pad <= mag_table({}) & mag_table & "{}";\n'.format(*tuple_val))
        tuple_val = (output_msb + addr_bits, addr_bits)
        fh.write('interp_out_slice <= interp_out({} downto {});\n'.format(*tuple_val))
        fh.write('\n')
        fh.write('shift_proc:\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('\n')
        fh.write('	    lower_slice_d(0) <= lower_slice;\n')
        fh.write('	    for i in 1 to 2 loop\n')
        fh.write('	        lower_slice_d(i) <= lower_slice_d(i-1);\n')
        fh.write('	    end loop;\n')
        fh.write('\n')
        fh.write('        upper_slice_d(0) <= upper_slice;\n')
        fh.write('        for i in 1 to 6 loop\n')
        fh.write('            upper_slice_d(i) <= upper_slice_d(i-1);\n')
        fh.write('        end loop;\n')
        fh.write('\n')
        fh.write('        small_table_out_d(0) <= mag_small_table;\n')
        fh.write('        for i in 1 to 3 loop\n')
        fh.write('            small_table_out_d(i) <= small_table_out_d(i-1);\n')
        fh.write('        end loop;\n')
        fh.write('        mux_out <= next_mux_out;\n')
        if tlast:
            fh.write('        tlast_d <= tlast_d(6 downto) & s_axis_tuser;\n')

        if tuser_width > 0:
            fh.write('        tuser_d0 <= s_axis_tuser;\n')
            fh.write('        tuser_d1 <= tuser_d0;\n')
            fh.write('        tuser_d2 <= tuser_d1;\n')
            fh.write('        tuser_d3 <= tuser_d2;\n')
            fh.write('        tuser_d4 <= tuser_d3;\n')
            fh.write('        tuser_d5 <= tuser_d4;\n')
            fh.write('        tuser_d6 <= tuser_d5;\n')
            fh.write('        tuser_d7 <= tuser_d6;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('valid_proc:\n')
        fh.write('process(clk,\n')
        fh.write('        sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('	    if (sync_reset = \'1\') then\n')
        fh.write('		    tvalid_d <= (others=>\'0\');\n')
        fh.write('	    else\n')
        fh.write('          tvalid_d <= tvalid_d(6 downto 0) & (s_axis_tvalid and (not almost_full));\n')
        fh.write('	    end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('--mux Process Latency 1.\n')
        fh.write('async_proc:\n')
        fh.write('process(small_table_out_d,\n')
        fh.write('        interp_out_slice,\n')
        fh.write('        upper_slice_d)\n')
        fh.write('begin\n')
        fh.write('	if ((upper_slice_d(6) = ALL_ONES) or (upper_slice_d(6) = ALL_ZEROS)) then\n')
        fh.write('		next_mux_out <= small_table_out_d(3);\n')
        fh.write('	else\n')
        fh.write('		next_mux_out <= interp_out_slice;\n')
        fh.write('	end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('-- Latency = 3.\n')
        fh.write('log_table : {}\n'.format(table_name))
        fh.write('  port map (\n')
        fh.write('      clk => clk,\n')
        fh.write('      addra => addra,\n')
        fh.write('      addrb => addrb,\n')
        fh.write('      doa => upper_table,\n')
        fh.write('      dob => lower_table\n')
        fh.write('  );\n')
        fh.write('\n')
        fh.write('-- Latency = 4.\n')
        fh.write('\n')
        fh.write('interp_dsp : {}\n'.format(dsp_name))
        fh.write('  port map (\n')
        fh.write('    clk => clk,\n')
        fh.write('    a => diff_table,\n')
        fh.write('    b => bterm,\n')
        fh.write('    c => large_table_pad,\n')
        fh.write('    p => interp_out\n')
        fh.write('  );\n')
        axi_fifo_inst(fh, fifo_name, inst_name='u_fifo', data_width=output_width, af_thresh=af_thresh,
                      addr_width=addr_width, tuser_width=tuser_width, tlast=tlast, s_tvalid_str='tvalid_d(7)',
                      s_tdata_str='mux_out', s_tuser_str='tuser_d7', s_tlast_str='tlast_d(7)',
                      s_tready_str='open', almost_full_str='almost_full', m_tvalid_str='m_axis_tvalid', m_tdata_str='m_axis_tdata',
                      m_tuser_str='m_axis_tuser', m_tlast_str='m_axis_tlast', m_tready_str='m_axis_tready')

        fh.write('end rtl;\n')

def gen_count_items(input_width, count_width, latency=2, path=None, c_str=None):
    """
        Generates the count items module.
    """

    mod_str = 'count_items_iw{}_cw{}'.format(input_width, count_width)
    input_msb = input_width - 1
    count_msb = count_width - 1
    lat_bits = fp_utils.ret_num_bitsU(latency - 1)
    reset_str = fp_utils.dec_to_ubin(latency - 1, num_bits=lat_bits)[0]
    count_name = 'counter_w{}_l{}'.format(count_width, latency)
    if path is not None:
        file_name = path + '/' + mod_str + '.vhd'
    else:
        file_name = './' + mod_str
    with open(file_name, "w") as fh:
        fh.write('--*************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : {}'.format(mod_str))
        fh.write('-- Description : Module time aligns a count sequence with the incoming data.\n')
        fh.write('--               Using a pipelined counter coure.\n')
        fh.write('--\n')
        fh.write('--\n')
        fh.write('--This software is property of Vallance Engineering, LLC and may\n')
        fh.write('--not be used, reviewed, or distributed without prior written consent.\n')
        fh.write('--                                                        (c) 2016\n')
        fh.write('--*************************************************************************--\n')
        fh.write('\n')
        fh.write('library ieee;\n')
        fh.write('use ieee.std_logic_1164.all;\n')
        fh.write('use ieee.numeric_std.all;\n')
        fh.write('\n')
        fh.write('library work;\n')
        fh.write('\n')
        fh.write('package {}_cmp is\n'.format(mod_str))
        fh.write('	component {}\n'.format(mod_str))
        fh.write('		port (\n')
        fh.write('			sync_reset : in std_logic;\n')
        fh.write('			clk : in std_logic;\n')
        fh.write('			valid_i : in std_logic;\n')
        fh.write('			reset_cnt : in std_logic;\n')
        fh.write('			data_i : in std_logic_vector({} downto 0);\n'.format(input_msb))
        fh.write('			valid_o : out std_logic;\n')
        fh.write('			count_o : out std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('			data_o : out std_logic_vector({} downto 0)\n'.format(input_msb))
        fh.write('		);\n')
        fh.write('	end component;\n')
        fh.write('\n')
        fh.write('end package {}_cmp;\n'.format(mod_str))
        fh.write('\n')
        fh.write('library ieee;\n')
        fh.write('use ieee.std_logic_1164.all;\n')
        fh.write('use ieee.numeric_std.all;\n')
        fh.write('\n')
        fh.write('entity {} is\n'.format(mod_str))
        fh.write('	port (\n')
        fh.write('		sync_reset : in std_logic;\n')
        fh.write('		clk : in std_logic;\n')
        fh.write('		valid_i : in std_logic;\n')
        fh.write('		reset_cnt : in std_logic;\n')
        fh.write('		data_i : in std_logic_vector({} downto 0);\n'.format(input_msb))
        fh.write('		valid_o : out std_logic;\n')
        fh.write('		count_o : out std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('		data_o : out std_logic_vector({} downto 0)\n'.format(input_msb))
        fh.write('	);\n')
        fh.write('end {};\n'.format(mod_str))
        fh.write('\n')
        fh.write('architecture rtl of {} is\n'.format(mod_str))
        fh.write('\n')
        fh.write('	signal cnt : std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('	constant l_value : std_logic_vector({} downto 0) := (others => \'0\');\n'.format(count_msb))
        fh.write('\n')
        for i in range(latency):
            fh.write('	signal out_d{}, next_out_d{} : std_logic_vector({} downto 0);\n'.format(i, i, input_msb))
        fh.write('\n')
        fh.write('	signal init_cnt, next_init_cnt : unsigned({} downto 0);\n'.format(lat_bits - 1))
        fh.write('	signal count_valid, next_count_valid : std_logic;\n')
        fh.write('    signal first_flag, next_first_flag : std_logic;\n')
        fh.write('    signal reset_cnt_s : std_logic;\n')
        fh.write('\n')
        fh.write('	COMPONENT {}\n'.format(count_name))
        fh.write('	  PORT (\n')
        fh.write('	    clk : IN STD_LOGIC;\n')
        fh.write('	    ce : IN STD_LOGIC;\n')
        fh.write('	    load : IN STD_LOGIC;\n')
        fh.write('        sclr : IN STD_LOGIC;\n')
        fh.write('	    l : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(count_msb))
        fh.write('	    q : OUT STD_LOGIC_VECTOR({} DOWNTO 0)\n'.format(count_msb))
        fh.write('	  );\n')
        fh.write('	END COMPONENT;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('	valid_o <= count_valid;\n')
        fh.write('	data_o <= out_d{};\n'.format(latency - 1))
        fh.write('	count_o <= cnt;\n')
        format_str = '#0{}b'.format(lat_bits)  #analysis:ignore
        # init_str = format(latency-1, format_str)[2:]
        fh.write('    reset_cnt_s <= \'1\' when (first_flag = \'1\' and valid_i = \'1\') else reset_cnt;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('	-- do a reset\n')
        fh.write('	process(clk, sync_reset)\n')
        fh.write('	begin\n')
        fh.write('        if (rising_edge(clk) ) then\n')
        fh.write('            if (sync_reset = \'1\') then\n')
        fh.write('                init_cnt <= "{}";\n'.format(reset_str))
        fh.write('                count_valid <= \'0\';\n')
        fh.write('                first_flag <= \'1\';\n')
        for i in range(latency):
            fh.write('                out_d{} <= (others => \'0\');\n'.format(i))
        fh.write('            else\n')
        for i in range(latency):
            fh.write('                out_d{} <= next_out_d{};\n'.format(i, i))
        fh.write('                init_cnt <= next_init_cnt;\n')
        fh.write('                first_flag <= next_first_flag;\n')
        fh.write('                count_valid <= next_count_valid;\n')
        fh.write('            end if;\n')
        fh.write('        end if;\n')
        fh.write('	end process;\n')
        fh.write('\n')
        fh.write('    async_proc : process(valid_i,\n')
        fh.write('                         data_i,\n')
        fh.write('                         first_flag,\n')
        for i in range(latency):
            fh.write('                         out_d{},\n'.format(i))
        fh.write('                         init_cnt)\n')
        fh.write('	begin\n')
        fh.write('\n')
        for i in range(latency):
            fh.write('		next_out_d{} <= out_d{};\n'.format(i, i))
        fh.write('         next_init_cnt <= init_cnt;\n')
        fh.write('         next_count_valid <= \'0\';\n')
        fh.write('         next_first_flag <= first_flag;\n')
        fh.write('\n')
        fh.write('		if (valid_i = \'1\') then\n')
        fh.write('			next_out_d0 <= data_i;\n')
        fh.write('               next_first_flag <= \'0\';\n')
        for i in range(1, latency):
            fh.write('			next_out_d{} <= out_d{};\n'.format(i, i - 1))
        fh.write('			if (init_cnt /= 0) then\n')
        fh.write('				next_init_cnt <= init_cnt - 1;\n')
        fh.write('			end if;\n')
        fh.write('\n')
        fh.write('			if (init_cnt = 0) and (valid_i = \'1\') then\n')
        fh.write('				next_count_valid <= \'1\';\n')
        fh.write('			end if;\n')
        fh.write('		end if;\n')
        fh.write('	end process;\n')
        fh.write('\n')
        fh.write('	U_{} : {}\n'.format(count_name, count_name))
        fh.write('	  port map (\n')
        fh.write('	    clk => clk,\n')
        fh.write('        sclr => sync_reset,\n')
        fh.write('	    ce => valid_i,\n')
        fh.write('	    load => reset_cnt_s,\n')
        fh.write('	    l => l_value,\n')
        fh.write('	    q => cnt\n')
        fh.write('	  );\n')
        fh.write('\n')
        fh.write('end rtl;\n')

    if c_str is not None:
        c_str.write('##################################################\n')
        c_str.write('{} Cores\n'.format(mod_str))
        c_str.write('##################################################\n')
        c_str.write('############################\n')
        c_str.write('Counter\n')
        c_str.write('Latency = {}\n'.format(latency))
        c_str.write('Block Name = {}\n'.format(count_name))
        c_str.write('Use CE\n')
        c_str.write('Use Load\n')
        c_str.write('Use Sync Reset\n')
        c_str.write('Loadable = True\n')
        c_str.write('Output Width = {}\n'.format(count_width))
        c_str.write('############################\n')

    return mod_str


def gen_count_cycle(input_width, count_width, latency=2, add_latency=1,
                    repeat=False, path=None, c_str=None):
    if repeat:
        mod_str = 'count_repeat_iw{}_cw{}'.format(input_width, count_width)
    else:
        mod_str = 'count_cycle_iw{}_cw{}'.format(input_width, count_width)
    input_msb = input_width - 1
    count_msb = count_width - 1
    # lat_bits = fp_utils.ret_num_bitsU(latency)
    adder_name = 'count_offset_w{}_l{}'.format(count_width, add_latency)
    count_items_name = gen_count_items(input_width, count_width, latency, path, c_str)
    if path is not None:
        file_name = path + '/' + mod_str + '.vhd'
    else:
        file_name = './' + mod_str
    with open(file_name, "w") as fh:

        fh.write('--**************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : {}.v\n'.format(mod_str))
        fh.write('-- Description : Implement a counter that is time aligned with the data input.\n')
        fh.write('--             : The reset of the counter is specifed by the input high_cnt.\n')
        fh.write('--\n')
        fh.write('-- This software is property of Vallance Engineering, LLC and may\n')
        fh.write('-- not be used, reviewed, or distributed without prior written consent.\n')
        fh.write('--                                                      (c) 2016\n')
        fh.write('--\n')
        fh.write('--**************************************************************************--\n')
        fh.write('\n')

        fh.write('library ieee;\n')
        fh.write('use ieee.std_logic_1164.all;\n')
        fh.write('use ieee.numeric_std.all;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('package {}_cmp is\n'.format(mod_str))
        fh.write('	component {}\n'.format(mod_str))
        fh.write('		port (\n')
        fh.write('			sync_reset : in std_logic;\n')
        fh.write('			clk : in std_logic;\n')
        fh.write('			valid_i : in std_logic;\n')
        fh.write('			high_cnt : in std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('			data_i : in std_logic_vector({} downto 0);\n'.format(input_msb))
        if repeat:
            fh.write('			ready_o : out std_logic;\n'.format(input_msb))
        fh.write('			valid_o : out std_logic;\n')
        fh.write('			count_o : out std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('			data_o : out std_logic_vector({} downto 0)\n'.format(input_msb))
        fh.write('		);\n')
        fh.write('	end component;\n')
        fh.write('\n')
        fh.write('end package {}_cmp;\n'.format(mod_str))
        fh.write('\n')
        fh.write('library ieee;\n')
        fh.write('use ieee.std_logic_1164.all;\n')
        fh.write('use ieee.numeric_std.all;\n')
        fh.write('\n')
        fh.write('library work;\n')
        fh.write('use work.{}_cmp.all;\n'.format(count_items_name))
        fh.write('\n')
        fh.write('entity {} is\n'.format(mod_str))
        fh.write('	port (\n')
        fh.write('		sync_reset : in std_logic;\n')
        fh.write('		clk : in std_logic;\n')
        fh.write('		valid_i : in std_logic;\n')
        fh.write('		high_cnt : in std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('		data_i : in std_logic_vector({} downto 0);\n'.format(input_msb))
        if repeat:
            fh.write('		ready_o : out std_logic;\n'.format(input_msb))
        fh.write('		valid_o : out std_logic;\n')
        fh.write('		count_o : out std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('		data_o : out std_logic_vector({} downto 0)\n'.format(input_msb))
        fh.write('	);\n')
        fh.write('end {};\n'.format(mod_str))
        fh.write('\n')
        fh.write('architecture rtl of {} is\n'.format(mod_str))
        fh.write('\n')
        fh.write('signal reset_cnt, next_reset_cnt : std_logic;\n')
        fh.write('signal count_value : std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('signal count_valid : std_logic;\n')
        fh.write('signal count_data : std_logic_vector({} downto 0);\n'.format(input_msb))
        fh.write('\n')
        fh.write('signal reset_flag1, reset_flag2 : std_logic_vector({} downto 0);\n'.format(count_msb))
        fh.write('\n')
        if repeat:
            fh.write('signal ready, next_ready : std_logic;\n')
            # fh.write('signal reset_cnt_s : std_logic;\n')
        fh.write('signal valid_d1 : std_logic;\n')
        fh.write('signal data_d1 : std_logic_vector({} downto 0);\n'.format(input_msb))
        fh.write('\n')
        fh.write('COMPONENT {}\n'.format(adder_name))
        fh.write('PORT (\n')
        fh.write('A : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(count_msb))
        fh.write('B : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(count_msb))
        fh.write('CLK : IN STD_LOGIC;\n')
        fh.write('S : OUT STD_LOGIC_VECTOR({} DOWNTO 0)\n'.format(count_msb))
        fh.write(');\n')
        fh.write('END COMPONENT;\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('valid_o <= count_valid;\n')
        fh.write('data_o <= count_data;\n')
        if repeat:
            fh.write('ready_o <= ready;\n')
        fh.write('count_o <= count_value;\n')
        fh.write('\n')
        if repeat:
            fh.write('process(clk, sync_reset)\n')
            fh.write('begin\n')
            fh.write('     if (rising_edge(clk)) then\n')
            fh.write('	    if (sync_reset = \'1\') then\n')
            fh.write('            reset_cnt <= \'1\';\n')
            fh.write('            ready <= \'0\';\n')
            fh.write('	    else\n')
            fh.write('            reset_cnt <= next_reset_cnt;\n')
            fh.write('            ready <= next_ready;\n')
            fh.write('         end if;\n')
            fh.write('	end if;\n')
            fh.write('end process;\n')
            fh.write('\n')
        else:
            fh.write('process(clk, sync_reset)\n')
            fh.write('begin\n')
            fh.write('     if (rising_edge(clk)) then\n')
            fh.write('	    if (sync_reset = \'1\') then\n')
            fh.write('             reset_cnt <= \'1\';\n')
            fh.write('	    else\n')
            fh.write('             reset_cnt <= next_reset_cnt;\n')
            fh.write('	    end if;\n')
            fh.write('	end if;\n')
            fh.write('end process;\n')
            fh.write('\n')
        fh.write('\n')
        fh.write('-- delay process\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        valid_d1 <= valid_i;\n')
        fh.write('        data_d1 <= data_i;\n')
        fh.write('	end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('-- write process.\n')
        fh.write('process(reset_cnt, valid_i, count_value, valid_d1, reset_flag2, reset_flag1)\n')
        fh.write('begin\n')
        if repeat:
            fh.write('    next_ready <= \'0\';\n')
        fh.write('    if (reset_cnt = \'1\') then\n')
        fh.write('        next_reset_cnt <= \'0\';\n')
        fh.write('    elsif (valid_i = \'1\') then\n')
        str_val = fp_utils.dec_to_ubin(1, count_width)[0]
        if repeat:
            fh.write('        if (count_value = reset_flag2) then\n')
            fh.write('            next_reset_cnt <= \'1\';\n')
            fh.write('            next_ready <= \'1\';\n')
        else:
            fh.write('        if (((count_value = reset_flag1) and (valid_d1 = \'0\')) or ((count_value = reset_flag2) and (valid_d1 = \'1\'))) then\n')
            fh.write('            next_reset_cnt <= \'1\';\n')
        fh.write('        else\n')
        fh.write('            next_reset_cnt <= \'0\';\n')
        fh.write('        end if;\n')
        fh.write('    else\n')
        fh.write('        next_reset_cnt <= \'0\';\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('count_offset2 : {}\n'.format(adder_name))
        fh.write('port map (\n')
        fh.write('  A => high_cnt,\n')
        str_val = fp_utils.dec_to_ubin(latency, count_width)[0]
        fh.write('  B => "{}",\n'.format(str_val))
        fh.write('  CLK => clk,\n')
        fh.write('  S => reset_flag2\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('count_offset1 : {}\n'.format(adder_name))
        fh.write('port map (\n')
        fh.write('  A => high_cnt,\n')
        str_val = fp_utils.dec_to_ubin(latency - 1, count_width)[0]
        fh.write('  B => "{}",\n'.format(str_val))
        fh.write('  CLK => clk,\n')
        fh.write('  S => reset_flag1\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('count_items : {}\n'.format(count_items_name))
        fh.write('port map (\n')
        fh.write('	sync_reset => sync_reset,\n')
        fh.write('	clk => clk,\n')
        fh.write('	valid_i => valid_d1,\n')
#        if repeat:
#            fh.write('	reset_cnt => reset_cnt_s,\n')
#        else:
        fh.write('	reset_cnt => reset_cnt,\n')
        fh.write('	data_i => data_d1,\n')
        fh.write('	valid_o => count_valid,\n')
        fh.write('	count_o => count_value,\n')
        fh.write('	data_o => count_data\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('end rtl;\n')

    if c_str is not None:
        c_str.write('##################################################\n')
        c_str.write('{} Cores\n'.format(mod_str))
        c_str.write('##################################################\n')
        c_str.write('############################\n')
        c_str.write('Subtract\n')
        c_str.write('Latency = {}\n'.format(add_latency))
        c_str.write('Block Name = {}\n'.format(adder_name))
        c_str.write('Unsigned Input A Width = {}\n'.format(count_width))
        c_str.write('Unsigned Input B Width = {}\n'.format(count_width))
        c_str.write('Use CE = False\n')
        c_str.write('Use Sync Reset = False\n')
        c_str.write('Output Width = {}\n'.format(count_width))
        c_str.write('############################\n')


def gen_axi_downsample(path, tlast=False, tuser_width=0):

    path = ret_valid_path(path)

    hash = 0
    if tlast:
        hash += 1
    if tuser_width > 0:
        hash += 2

    mod_name = 'axi_downsample_{}'.format(hash)
    file_name = name_help(mod_name, path)
    module_name = ret_module_name(file_name)

    tuser_msb = tuser_width - 1
    with open(file_name, 'w') as fh:

        fh.write('\n')
        fh.write('--***************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : downsample.v\n')
        fh.write('-- Description : Module used for variable decimation of a sample stream.\n')
        fh.write('--\n')
        fh.write('--***************************************************************************--\n')
        fh.write('\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        print_intro(fh, module_name)
        fh.write('        generic\n')
        fh.write('        (\n')
        fh.write('            DATA_WIDTH : integer := 32;\n')
        if tuser_width:
            fh.write('            TUSER_WIDTH : integer := 8;\n')
        fh.write('            CNT_BITS : integer := 6\n')
        fh.write('        );\n')
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('            rate : in std_logic_vector(CNT_BITS-1 downto 0);\n')
        fh.write('\n')
        fh.write('            s_axis_tvalid : in std_logic;\n')
        if tlast:
            fh.write('            s_axis_tlast : in std_logic;\n')
        if tuser_width > 0:
            fh.write('            s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('            s_axis_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0),\n')
        fh.write('            s_axis_tready : out std_logic;\n')
        fh.write('\n')
        fh.write('            m_axis_tvalid : out std_logic;\n')
        if tlast:
            fh.write('            m_axis_tlast : out std_logic;\n')
        if tuser_width > 0:
            fh.write('            m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('            m_axis_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        fh.write('            m_axis_tready : in std_logic\n')
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    generic\n')
        fh.write('    (\n')
        fh.write('        DATA_WIDTH : integer := 32;\n')
        if tuser_width:
            fh.write('        TUSER_WIDTH : integer := 8;\n')
        fh.write('        CNT_BITS : integer := 6\n')
        fh.write('    );\n')
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('        rate : in std_logic_vector(CNT_BITS-1 downto 0);\n')
        fh.write('\n')
        fh.write('        s_axis_tvalid : in std_logic;\n')
        if tlast:
            fh.write('        s_axis_tlast : in std_logic;\n')
        if tuser_width > 0:
            fh.write('        s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('        s_axis_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0),\n')
        fh.write('        s_axis_tready : out std_logic;\n')
        fh.write('\n')
        fh.write('        m_axis_tvalid : out std_logic;\n')
        if tlast:
            fh.write('        m_axis_tlast : out std_logic;\n')
        if tuser_width > 0:
            fh.write('        m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('        m_axis_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        fh.write('        m_axis_tready : in std_logic\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n\n'.format(module_name))
        fh.write('signal cnt, next_cnt : unsigned(CNT_BITS-1 downto 0);\n')
        fh.write('constant ZERO_CNT : unsigned(CNT_BITS-1 downto 0) := (others => \'0\');\n')
        fh.write('signal rate_s : unsigned(CNT_BITS-1 downto 0);\n')
        fh.write('signal signal_out, next_signal_out : std_logic_vector(DATA_WIDTH-1 downto 0);\n')
        fh.write('signal rdyfordata : std_logic;\n')
        if tlast:
            fh.write('signal tlast_out, next_tlast_out : std_logic;\n')
        if tuser_width > 0:
            fh.write('signal tuser_out, next_tuser_out : std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        fh.write('\n')
        fh.write('signal valid_s, next_valid_s : std_logic;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('m_axis_tvalid <= valid_s;\n')
        fh.write('m_axis_tdata <= signal_out;\n')
        fh.write('s_axis_tready <= rdyfordata;\n')
        if tlast:
            fh.write('m_axis_tlast <= tlast_out;\n')
        if tuser_width > 0:
            fh.write('m_axis_tuser <= tuser_out;\n')

        fh.write('rdyfordata = \'1\' when (m_axis_tready or not valid_s) else \'0\';\n')
        fh.write('\n')
        fh.write('-- main clock process\n')
        fh.write('proc_clk : process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        fh.write('            valid_s <= \'0\';\n')
        fh.write('            signal_out <= (others => \'0\');\n')
        if tlast:
            fh.write('            tlast_out <= \'0\';\n')
        if tuser_width > 0:
            fh.write('            tuser_out <= (others => \'0\');\n')
        fh.write('            rate_s <= unsigned(rate_i) - 1;\n')
        fh.write('            cnt     <= (others => \'0\');\n')
        fh.write('        else\n')
        fh.write('            valid_s <= next_valid_s;\n')
        fh.write('            rate_s <= unsigned(rate_i) - 1;\n')
        fh.write('            signal_out <= next_signal_out;\n')
        if tlast:
            fh.write('            tlast_out <= next_tlast_out;\n')
        if tuser_width > 0:
            fh.write('            tuser_out <= next_tuser_out;\n')
        fh.write('            cnt  <= next_cnt;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('--Async logic\n')
        fh.write('async_logic:\n')
        fh.write('process(signal_out,\n')
        if tlast:
            fh.write('        tlast_out,\n')
            fh.write('        s_axis_tlast,\n')
        if tuser_width > 0:
            fh.write('        tuser_out,\n')
            fh.write('        s_axis_tuser,\n')
        fh.write('        cnt,\n')
        fh.write('        s_axis_tvalid,\n')
        fh.write('        rdyfordata,\n')
        fh.write('        s_axis_tdata,\n')
        fh.write('        rate_s,\n')
        fh.write('        m_axis_tready)\n')
        fh.write('begin\n')
        fh.write('    next_valid_s = \'0\';\n')
        fh.write('    next_signal_out = signal_out;\n')
        fh.write('    next_cnt = cnt;\n')
        if tlast:
            fh.write('    next_tlast_out = tlast_out;\n')
        if tuser_width > 0:
            fh.write('    next_tuser_out = tuser_out;\n')
        fh.write('    if (s_axis_tvalid = \'1\' and rdyfordata = \'1\') then\n')
        fh.write('        if (cnt = ZERO_CNT) then\n')
        fh.write('            next_valid_s <= \'1\';\n')
        fh.write('            next_signal_out <= s_axis_tdata;\n')
        if tlast:
            fh.write('            next_tlast_out <= s_axis_tlast;\n')
        if tuser_width > 0:
            fh.write('            next_tuser_out <= s_axis_tuser;\n')
        fh.write('        end if;\n')
        fh.write('        if (cnt = rate_s) then\n')
        fh.write('            next_cnt <= 0;\n')
        fh.write('        else\n')
        fh.write('            next_cnt <= cnt + 1;\n')
        fh.write('        end if;\n')
        fh.write('    elsif (m_axis_tready = \'1\') then\n')
        fh.write('        next_valid_s = \'0\';\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('end rtl;\n')

    return module_name

def gen_var_delay(file_dir, depth, width, c_str=None):

    # device='xc6slx45t',
    # family='spartan6',
    # package='csg324',
    # speed_grade=3):

    m_setting_bits = int(np.ceil(np.log2(depth)))
    depth = 2**m_setting_bits

    file_name = 'var_delay_%dw_%dd.vhd' % (width, depth)
    file_name = os.path.join(file_dir, file_name)
    with open(file_name, 'w') as fh:
        module_name = ret_module_name(file_name)

        m_setting_msb = m_setting_bits - 1
        msb = width - 1

        # generate sub modules.
    #    proj_dir = os.path.dirname(os.path.realpath(project_file)) + '/'
    #    # generate cores
        # top_name = ret_file_name(file_name[:-2])

        cnt_name = 'cnt_{}w'.format(m_setting_bits)
        addr_name = 'subtract_{}w'.format(m_setting_bits)
        ram_name = 'ram_{}w_{}d'.format(width, depth)
        cnt_msb = m_setting_bits - 1

        fh.write('\n')
        fh.write('--*************************************************************'
                 '*************--\n')
        fh.write('--\n')
        fh.write('-- Author      : Phil Vallance\n')
        fh.write('-- File        : %s.v\n' % module_name)
        fh.write('-- Description : Implements a variable shift register using '
                 'block ram.\n')
        fh.write('--\n')
        print_header(fh)
        fh.write('--\n')
        fh.write('--*************************************************************'
                 '*************--\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('package {}_cmp is\n'.format(module_name))
        fh.write('    component {}\n'.format(module_name))
        fh.write('        port \n')
        fh.write('        ( \n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('\n')
        fh.write('            msetting : in std_logic_vector({} downto 0);\n'.format(m_setting_msb))
        fh.write('\n')
        fh.write('            valid_i : in std_logic;\n')
        fh.write('            signal_i : in std_logic_vector({} downto 0);\n'.format(msb))
        fh.write('\n')
        fh.write('            valid_o : out std_logic;\n')
        fh.write('            signal_o : out std_logic_vector({} downto 0)\n'.format(msb))
        fh.write('        );\n')
        fh.write('    end component;\n')
        fh.write('end package {}_cmp;\n'.format(module_name))
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    port \n')
        fh.write('    ( \n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('\n')
        fh.write('        msetting : in std_logic_vector({} downto 0);\n'.format(m_setting_msb))
        fh.write('\n')
        fh.write('        valid_i : in std_logic;\n')
        fh.write('        signal_i : in std_logic_vector({} downto 0);\n'.format(msb))
        fh.write('\n')
        fh.write('        valid_o : out std_logic;\n')
        fh.write('        signal_o : out std_logic_vector({} downto 0)\n'.format(msb))
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is\n'.format(module_name))
        fh.write('\tsignal wr_addr_d1, wr_addr_d2 : std_logic_vector({} downto 0);\n'.format(m_setting_msb))  #analysis:ignore
        fh.write('\tsignal rd_addr : std_logic_vector({} downto 0);\n'.format(m_setting_msb))
        fh.write('\tsignal valid_d1, valid_d2, valid_d3, valid_d4, valid_d5, valid_d6 : std_logic;\n')  #analysis:ignore
        fh.write('\tsignal wr_addr : std_logic_vector({} downto 0);\n'.format(m_setting_msb))
        fh.write('\tsignal val_d1, val_d2, val_d3 : std_logic_vector({} downto 0);\n'.format(msb))
        fh.write('\tsignal we : std_logic_vector(0 downto 0);\n')
        fh.write('\n')
        fh.write('\t-- latency =3\n')
        fh.write('\tCOMPONENT ram_{}w_{}d\n'.format(width, depth))
        fh.write('\t  PORT (\n')
        fh.write('\t    clka : IN STD_LOGIC;\n')
        fh.write('\t    wea : IN STD_LOGIC_VECTOR(0 DOWNTO 0);\n')
        fh.write('\t    addra : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(cnt_msb))
        fh.write('\t    dina : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(msb))
        fh.write('\t    clkb : IN STD_LOGIC;\n')
        fh.write('\t    addrb : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(cnt_msb))
        fh.write('\t    doutb : OUT STD_LOGIC_VECTOR({} DOWNTO 0)\n'.format(msb))
        fh.write('\t  );\n')
        fh.write('\tEND COMPONENT;\n')
        fh.write('\n')
        fh.write('\t-- latency = 2\n')
        fh.write('\tCOMPONENT {}\n'.format(addr_name))
        fh.write('\t  PORT (\n')
        fh.write('\t    a : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(cnt_msb))
        fh.write('\t    b : IN STD_LOGIC_VECTOR({} DOWNTO 0);\n'.format(cnt_msb))
        fh.write('\t    clk : IN STD_LOGIC;\n')
        fh.write('\t    s : OUT STD_LOGIC_VECTOR({} DOWNTO 0)\n'.format(cnt_msb))
        fh.write('\t  );\n')
        fh.write('\tEND COMPONENT;\n')
        fh.write('\n')
        fh.write('\t-- latency = 2\n')
        fh.write('\tCOMPONENT {}\n'.format(cnt_name))
        fh.write('\t  PORT (\n')
        fh.write('\t    clk : IN STD_LOGIC;\n')
        fh.write('\t    ce : IN STD_LOGIC;\n')
        fh.write('\t    sclr : IN STD_LOGIC;\n')
        fh.write('\t    q : OUT STD_LOGIC_VECTOR({} DOWNTO 0)\n'.format(cnt_msb))
        fh.write('\t  );\n')
        fh.write('\tEND COMPONENT;\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('\tvalid_o <= valid_d6;\n')
        fh.write('\twe(0) <= valid_d2;\n')
        fh.write('\n')
        fh.write('sync_proc1:\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        val_d1 <= signal_i;\n')
        fh.write('        val_d2 <= val_d1;\n')
        fh.write('        val_d3 <= val_d2;\n')
        fh.write('        wr_addr_d1 <= wr_addr;\n')
        fh.write('        wr_addr_d2 <= wr_addr_d1;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('sync_proc2:\n')
        fh.write('process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('	\tif (sync_reset = \'1\') then\n')
        fh.write('	    \tvalid_d1 <= \'0\';\n')
        fh.write('        \tvalid_d2 <= \'0\';\n')
        fh.write('	    \tvalid_d3 <= \'0\';\n')
        fh.write('        \tvalid_d4 <= \'0\';\n')
        fh.write('	    \tvalid_d5 <= \'0\';\n')
        fh.write('        \tvalid_d6 <= \'0\';\n')
        fh.write('	\telse\n')
        fh.write('	    \tvalid_d1 <= valid_i;\n')
        fh.write('        \tvalid_d2 <= valid_d1;\n')
        fh.write('	    \tvalid_d3 <= valid_d2;\n')
        fh.write('        \tvalid_d4 <= valid_d3;\n')
        fh.write('	    \tvalid_d5 <= valid_d4;\n')
        fh.write('        \tvalid_d6 <= valid_d5;\n')
        fh.write('	\tend if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n')
        fh.write('\n')
        fh.write('address_gen : {}\n'.format(cnt_name))
        fh.write('  port map (\n')
        fh.write('    clk => clk,\n')
        fh.write('    sclr => sync_reset,\n')
        fh.write('    ce => valid_i,\n')
        fh.write('    q => wr_addr\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('address_offset : {}\n'.format(addr_name))
        fh.write('  port map (\n')
        fh.write('    a => wr_addr,\n')
        fh.write('    b => msetting, \n')
        fh.write('    clk => clk, \n')
        fh.write('    s => rd_addr\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('shift_memory : {}\n'.format(ram_name))
        fh.write('  port map (\n')
        fh.write('    clka => clk, \n')
        fh.write('    wea => we,\n')
        fh.write('    addra => wr_addr_d2,\n')
        fh.write('    dina => val_d2,\n')
        fh.write('    clkb => clk,\n')
        fh.write('    addrb => rd_addr,\n')
        fh.write('    doutb => signal_o\n')
        fh.write(');\n')
        fh.write('\n')
        fh.write('end rtl;\n')
        fh.close()

        if c_str is not None:
            c_str.write('##################################################\n')
            c_str.write('{} Cores\n'.format(module_name))
            c_str.write('##################################################\n')
            c_str.write('############################\n')
            c_str.write('Simple Dual Port RAM \n')
            c_str.write('Latency = 3\n')
            c_str.write('Block Name = %s\n' % ram_name)
            c_str.write('Depth = {}\n'.format(depth))
            c_str.write('Width = {}\n'.format(width))
            c_str.write('############################\n')
            c_str.write('Subtraction\n')
            c_str.write('Block Name = %s\n' % addr_name)
            c_str.write('Latency = 2\n')
            c_str.write('Unsigned Input A Width = %d\n' % m_setting_bits)
            c_str.write('Unsigned Input B Width = %d\n' % m_setting_bits)
            c_str.write('No CE\n')
            c_str.write('No SCLR\n')
            c_str.write('Output Width = %d\n' % m_setting_bits)
            c_str.write('############################\n')
            c_str.write('Counter\n')
            c_str.write('Latency = 2\n')
            c_str.write('Block Name = %s\n' % cnt_name)
            c_str.write('Use CE\n')
            c_str.write('Use SCLR\n')
            c_str.write('Output Width = %d\n' % m_setting_bits)
            c_str.write('############################\n')


def gen_shifter(file_name, project_file, input_width, shift_bits, gain_width, output_width=None, device='xc6slx45t',
                family='spartan6', package='csg324', speed_grade=3):

    assert(file_name is not None), 'User must specify File Name'
    fh = open(file_name, 'w')
    module_name = ret_module_name(file_name)

    if (output_width is None):
        output_width = input_width
    slice_bits = input_width + gain_width - 1

    offset = file_name.rfind('.')
    shifter_name = ret_file_name(file_name[:offset])
    proj_dir = os.path.dirname(os.path.realpath(project_file)) + '/'
    curr_dir = os.path.dirname(os.path.realpath(file_name))

    gain_mult_name = shifter_name + '_Gain'
    gain_mult_file = proj_dir + gain_mult_name + '.xco'

    mult_width = gain_width + input_width

    shift_msb = shift_bits - 1
    mult_msb = mult_width - 1
    output_msb = output_width - 1
    input_msb = input_width - 1
    gain_msb = gain_width - 1

    fh.write('\n')
    fh.write('/--**************************************************************************--\n')
    fh.write('--\n')
    fh.write('-- Author      : Phil Vallance\n')
    fh.write('-- File        : {}.v\n'.format(module_name))
    fh.write('-- Description : Module performs scaling and bit slice as part of an gain block\n')
    fh.write('--\n')
    print_header(fh)
    fh.write('\n')
    fh.write('--\n')
    fh.write('/--**************************************************************************--\n')
    fh.write('\n')
    fh.write('\n')
    fh.write('module {}\n'.format(module_name))
    fh.write('(\n')
    fh.write('  clk : in std_logic; -- clock\n')
    fh.write('  sync_reset : in sync_reset; -- reset\n')
    fh.write('\n')
    fh.write('  s_axis_tvalid : in std_logic;\n')
    fh.write('  input [{}:0] mult_factor_i,\n'.format(gain_msb))
    fh.write('  input [%d:0] shift_factor_i,\n' % shift_msb)
    fh.write('\n')
    fh.write('  input [%d:0] i_input,\n' % input_msb)
    fh.write('  input [%d:0] q_input,\n' % input_msb)
    fh.write('\n')
    fh.write('  output valid_o,\n')
    fh.write('  output [%d:0] i_output,\n' % output_msb)
    fh.write('  output [%d:0] q_output,\n' % output_msb)
    fh.write('  output overflow_o\n')
    fh.write('\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('parameter BLOCK_LATENCY = 6;\n')
    fh.write('\n')
    fh.write('signal [%d:0] i_gain, q_gain;\n' % mult_msb)
    fh.write('\n')
    fh.write('signal [%d:0] shift_val_d [3:0];\n' % shift_msb)
    fh.write('\n')
    fh.write('signal [4:0] i_sign, q_sign;\n')
    fh.write('\n')
    fh.write('signal [BLOCK_LATENCY-1:0] valid_d;\n')
    fh.write('\n')
    fh.write('signal [%d:0] i_shift, q_shift;\n' % output_msb)
    str_val = 'i_output, next_i_output, q_output, next_q_output'
    reg_str = 'signal [%d:0] ' % output_msb
    fh.write(reg_str + str_val + ';\n')
    fh.write('signal next_i_overflow, i_overflow, next_q_overflow, q_overflow;\n')
    fh.write('\n')
    fh.write('signal [%d:0] i_gain_slice, q_gain_slice;\n' % (mult_msb - 1))
    fh.write('\n')
    fh.write('begin\n')
    fh.write('\n')
    fh.write('i_gain_slice <= i_gain({} downto 0);\n'.format(mult_msb - 1))
    fh.write('q_gain_slice <= q_gain({} downto 0);\n'.format(mult_msb - 1))
    fh.write('\n')
    fh.write('valid_o <= valid_d(BLOCK_LATENCY-1);\n')
    # fh.write('i_output = i_output;\n')
    # fh.write('q_output = q_output;\n')
    fh.write('overflow_o = i_overflow | q_overflow;\n')
    fh.write('\n')
    fh.write('integer ii;\n')
    fh.write('\n')
    fh.write('process(clk, sync_reset)\n')
    fh.write('begin\n')
    fh.write('	  if (sync_reset = \'1\') then\n')
    fh.write('        valid_d  <= 0;\n')
    fh.write('        i_overflow <= \'0\';\n')
    fh.write('        q_overflow <= \'0\';\n')
    fh.write('	  else\n')
    fh.write('		valid_d <= {valid_d[BLOCK_LATENCY-2:0],valid_i};\n')
    fh.write('        i_overflow <= next_i_overflow;\n')
    fh.write('        q_overflow <= next_q_overflow;\n')
    fh.write('	  end if;\n')
    fh.write('end process;\n')
    fh.write('\n')
    fh.write('process(clk)\n')
    fh.write('begin\n')
    fh.write('  i_sign[0] <= i_input[%d];\n' % input_msb)
    fh.write('  q_sign[0] <= q_input[%d];\n' % input_msb)
    fh.write('	shift_val_d[0] <= shift_factor_i;\n')
    fh.write('  for (ii=1; ii<5; ii=ii+1) then\n')
    fh.write('    q_sign[ii] <= q_sign[ii-1];\n')
    fh.write('    i_sign[ii] <= i_sign[ii-1];\n')
    fh.write('  end if;\n')
    fh.write('	for (ii = 1; ii < 4; ii = ii + 1) then\n')
    fh.write('		  shift_val_d[ii] <= shift_val_d[ii-1];\n')
    fh.write('	end if;\n')
    fh.write('  i_output <= next_i_output;\n')
    fh.write('  q_output <= next_q_output;\n')
    fh.write('end if;\n')
    fh.write('\n')
    fh.write('--mux Process Latency 1.\n')
    fh.write('always @*\n')
    fh.write('begin\n')
    fh.write('  next_i_overflow = \'0\';\n')
    fh.write('  next_q_overflow = \'0\';\n')
    fh.write('  if (i_sign[4] != i_shift[%d]) then\n' % output_msb)
    fh.write('      next_i_overflow = \'1\';\n')
    fh.write('      if (i_sign[4] = \'1\') then\n')
    str0 = '1' + (output_width - 1) * '0'
    fh.write('        next_i_output = %d\'b%s;\n' % (output_width, str0))
    fh.write('      else\n')
    str1 = '0' + (output_width - 1) * '1'
    fh.write('        next_i_output = %d\'b%s;\n' % (output_width, str1))
    fh.write('      end if;\n')
    fh.write('  else\n')
    fh.write('    next_i_output = i_shift;\n')
    fh.write('  end if;\n')
    fh.write('  if (q_sign[4] != q_shift[%d]) then\n' % output_msb)
    fh.write('    next_q_overflow = \'0\';\n')
    fh.write('    if (q_sign[4] = \'1\') then\n')
    fh.write('      next_q_output = %d\'b%s;\n' % (output_width, str0))
    fh.write('    else\n')
    fh.write('      next_q_output = %d\'b%s;\n' % (output_width, str1))
    fh.write('    end if;\n')
    fh.write('  else\n')
    fh.write('    next_q_output = q_shift;\n')
    fh.write('  end if;\n')
    fh.write('end if;\n')
    fh.write('\n')
    fh.write('-- Latency = 4.\n')
    fh.write('%s Gain_I (\n' % gain_mult_name)
    fh.write('  .clk(clk), \n')
    fh.write('  .a(i_input),\n')
    fh.write('  .b(mult_factor_i),\n')
    fh.write('  .p(i_gain)\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('-- Latency = 4.\n')
    fh.write('%s Gain_Q (\n' % gain_mult_name)
    fh.write('  .clk(clk),\n')
    fh.write('  .a(q_input),\n')
    fh.write('  .b(mult_factor_i),\n')
    fh.write('  .p(q_gain)\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('slicer%d_%d Shift_I (\n' % (slice_bits, output_width))
    fh.write('  .sync_reset(sync_reset), -- reset\n')
    fh.write('  .clk(clk), -- clock\n')
    fh.write('\n')
    fh.write('-- Settings offet the slicer from the base value.\n')
    fh.write('  .slice_offset_i(shift_val_d[3]),\n')
    fh.write('\n')
    fh.write('  .valid_i(valid_d[3]),\n')
    fh.write('  .signal_i(i_gain_slice),\n')
    fh.write('\n')
    fh.write('  .valid_o(shift_valid),\n')
    fh.write('  .signal_o(i_shift)\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('slicer%d_%d Shift_Q (\n' % (slice_bits, output_width))
    fh.write('  .sync_reset(sync_reset), -- reset\n')
    fh.write('  .clk(clk), -- clock\n')
    fh.write('\n')
    fh.write('-- Settings offet the slicer from the base value.\n')
    fh.write('  .slice_offset_i(shift_val_d[3]),\n')
    fh.write('\n')
    fh.write('  .valid_i(valid_d[3]),\n')
    fh.write('  .signal_i(q_gain_slice),\n')
    fh.write('\n')
    fh.write('  .valid_o(),\n')
    fh.write('  .signal_o(q_shift)\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('endmodule\n')
    fh.close()

    x_utils.xco_mult(gain_mult_file, project_file, device=device, family=family,
                     package=package,
                     speed_grade=speed_grade,
                     input_a_width=input_width,
                     input_b_width=gain_width,
                     a_type='Signed',
                     b_type='Unsigned',
                     pipe_stages=4,
                     mult_type='Parallel_Multiplier',
                     multiplier_construction='Use_Mults')

    gen_slicer(slice_bits, output_width, max_offset=2**shift_bits-1, file_path=curr_dir, rev_dir=True)


def gen_agc_files(agc_top_file, project_file, input_qvec, ki_max, log_width, exp_width, output_width,agc_ref,
                  input_type='Signed', filter_len=32, iterations=5, precision=0, sample_period=12, device='xc6slx45t',
                  family='spartan6', package='csg324', speed_grade=3):

    agc_obj = AGCModule(alpha_overflow=ki_max)

    signal_fi = fp_utils.sfi(0, input_qvec[0], input_qvec[1])
    # cordic uses 1 sign and 1 integer bits for representation

    (log_combined_fi, mult_value, exp_combined_fi,
     exp_table, exp_s_table,
     ls_table, mag_table,
     mag_s_table) = agc_obj.gen_fix_tables(signal_fi, equal_lens=True, log_width=log_width, exp_width=exp_width)

    agc_log_ref_fi = fp_utils.sfi(np.log(agc_ref), numeric_type=mag_table.numeric_type)

    # agc_log_ref = agc_log_ref_fi.udec
    shift_bits = ls_table.word_length
    input_width = input_qvec[0]

    filter_shift = int(np.ceil(np.log2(filter_len)))
    corr_factor = mult_value - 1

    word_len = log_width - 2
    corr_fac_fi = fp_utils.ufi(corr_factor, word_len, word_len)

    frac_width = log_width - 2
    frac_fi = fp_utils.ufi(0, frac_width, frac_width)
    corr_mult_fi = fp_utils.mult_fi(corr_fac_fi, frac_fi)

    ab_fi = fp_utils.mult_fi(exp_table, corr_mult_fi)
    ab_c_fi = fp_utils.add_fi(ab_fi, exp_table) #TODO: Do I need this?

    gain_out_fi = fp_utils.mult_fi(signal_fi, exp_table)
    max_shift = np.max(ls_table.dec)
    max_shift_bits = int(np.ceil(np.log2(max_shift)))
    max_shift = 2**max_shift_bits - 1

    curr_dir = os.path.dirname(os.path.realpath(agc_top_file))
    gen_slicer(gain_out_fi.word_length-1, signal_fi.word_length, max_offset=max_shift, file_path=curr_dir, rev_dir=True)

    d_a = copy.deepcopy(mag_table)
    # Note DSP48 internal structure does not add extra integer bit.
    # fp_utils.add_fi(mag_table, mag_table)
    ki_fi = fp_utils.sfi(agc_obj.alpha_overflow, 18)
    # must be signed due to operation of dsp macro.
    d_a_mult_ki_fi = fp_utils.mult_fi(d_a, ki_fi)
    loop_slice_bits = d_a_mult_ki_fi.fraction_length + mag_table.int_length
    loop_slice_msb = loop_slice_bits - 1

    file_name = agc_top_file
    assert(file_name is not None), 'User must specify File Name'
    fh = open(file_name, 'w')
    module_name = ret_module_name(file_name)

    offset = agc_top_file.rfind('.')
    agc_name = ret_file_name(agc_top_file[:offset])
    proj_dir = os.path.dirname(os.path.realpath(project_file)) + '/'
    cordic_name = agc_name + '_Abs'
    cordic_file = proj_dir + cordic_name + '.xco'

    filter_name = agc_name + '_Fil'
    filter_file = proj_dir + filter_name + '.xco'

    int_name = agc_name + '_integrator'
    int_file = proj_dir + int_name + '.xco'

    log_name = agc_name + '_log_conv'
    log_file = curr_dir + '/' + log_name + '.v'

    exp_name = agc_name + '_exp_conv'
    exp_file = curr_dir + '/' + exp_name + '.v'

    shifter_name = agc_name + '_shifter'
    shifter_file = curr_dir + '/' + shifter_name + '.v'

    coef_vals = np.ones((filter_len,))

    coef_vals_fi = fp_utils.ufi(coef_vals, 1, 0)

    fil_in_width = x_utils.port_width(input_width)

    fil_out_width = (fil_in_width +
                     int(np.ceil(np.log2(np.sum(coef_vals_fi.dec)))))

    fil_out_width = x_utils.port_width(fil_out_width)

    shift_msb = max_shift_bits - 1
    input_msb = input_width - 1
    output_msb = output_width - 1
    log_msb = log_width - 1
    exp_msb = exp_width - 1
    lsb = input_width - output_width

    fh.write('\n')
    fh.write('/--*************************************************************'
             '*************--\n')
    fh.write('--\n')
    fh.write('-- Author      : Phil Vallance\n')
    fh.write('-- File        : %s.v\n' % module_name)
    fh.write('-- Description : Module converts an input Magnitude to a log '
             'Value.  Note that\n')
    fh.write('--               this is a Natural log conversion '
             '(better compression of the\n')
    fh.write('--               signal.  The module uses linear interpolation '
             'to reduce table\n')
    fh.write('--               size and improve accuracy.\n')
    fh.write('--\n')
    fh.write('--\n')
    print_header(fh)
    fh.write('\n')
    fh.write('--\n')
    fh.write('/--*************************************************************'
             '*************--\n')
    fh.write('\n')
    fh.write('\n')
    fh.write('module %s\n' % module_name)
    fh.write('(\n')
    fh.write('  input          sync_reset, -- reset\n')
    fh.write('  input          clk, -- clock\n')
    fh.write('\n')
    fh.write('  input          valid_i,\n')
    fh.write('\n')
    fh.write('  input [17:0] k_i,\n')
    fh.write('  input [17:0] k_i_overflow,\n')
    fh.write('  input [%d:0] log_ref_i,\n' % log_msb)
    fh.write('\n')
    fh.write('  input [%d:0] i_input,\n' % input_msb)
    fh.write('  input [%d:0] q_input,\n' % input_msb)
    fh.write('\n')
    fh.write('  output valid_o,\n')
    fh.write('  output [%d:0] i_output,\n' % output_msb)
    fh.write('  output [%d:0] q_output\n' % output_msb)
    fh.write('\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('parameter LOOP_INT_LATENCY = 6;\n')
    fh.write('\n')
    fh.write('\n')
    fh.write('signal reset_n, reset_n_d1;\n')
    fh.write('signal areset;\n')
    fh.write('\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_reset_shift_d1 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_reset_shift_d2 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_reset_shift_d3 = \'1\';\n')
    fh.write('\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_reset_log_d1 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_reset_log_d2 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_reset_log_d3 = \'1\';\n')
    fh.write('\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_resetExp_d1 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_resetExp_d2 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_resetExp_d3 = \'1\';\n')
    fh.write('\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_resetLoop_d1 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_resetLoop_d2 = \'1\';\n')
    fh.write('(* KEEP = "TRUE" *) reg sync_resetLoop_d3 = \'1\';\n')
    fh.write('\n')
    num_bytes = int(np.ceil(input_width/8.))
    cord_bits = 2 * num_bytes * 8
    fh.write('signal [%d:0] CordicInput, CordicOutput;\n' % (cord_bits - 1))
    fh.write('signal CordicStb, log_valid, ExpStb, shift_valid, fil_valid;\n')
    fh.write('signal [%d:0] shift_i, shift_q;\n' % input_msb)
    fh.write('signal [%d:0] Mag;\n' % input_msb)
    fh.write('signal [%d:0] lin_val;\n' % exp_msb)
    fh.write('signal [%d:0] log_value;\n' % log_msb)
    fh.write('signal [%d:0] shift_val;\n' % shift_msb)
    fh.write('signal [47:0] loop_val;\n')
    fh.write('signal [%d:0] loop_val_slice;\n' % log_msb)
    fh.write('signal overflow_fil, overflow_cordic;\n')
    fil_bits = num_bytes * 8 + filter_shift
    fil_bits = x_utils.port_width(fil_bits)
    fh.write('signal [%d:0] fil_data;\n' % (fil_bits - 1))
    fh.write('signal overflow;\n')
    fh.write('\n')
    fh.write('signal [%d:0] loop_val_slice_d, next_loop_val_slice_d;\n' % log_msb)
    fh.write('\n')
    fh.write('signal [%d:0] shift_val_r, next_shift_val_r;\n' % shift_msb)
    fh.write('signal [%d:0] mult_factor_r, next_mult_factor_r;\n' % exp_msb)
    fh.write('\n')
    fh.write('signal [17:0] k_i_d, next_k_i_d;\n')
    fh.write('signal [%d:0] log_ref_d, next_log_ref_d, log_value_d,'
             ' next_log_value_d = 0;\n' % log_msb)
    fh.write('\n')
    fh.write('signal [LOOP_INT_LATENCY:0] log_valid_d; - + 1 due to '
             'latency of overflow check.\n')
    fh.write('signal loop_overflow, next_loop_overflow;\n')

    fh.write('\n')
    fh.write('signal [47:0] CVal, next_CVal;\n')
    fh.write('\n')
    frac_8 = input_width % 8
    if (frac_8 != 0):
        pad = 8 - frac_8
        temp = str(pad) + '\'b' + pad * '0'
        str_val = '{{},shift_q,{},shift_i}'.format(temp, temp)
    else:
        str_val = '{shift_q,shift_i}'
    fh.write('assign CordicInput = %s;\n' % str_val)
    fh.write('assign Mag = fil_data[%d:%d];\n'
             % (input_width + filter_shift - 1, filter_shift))
    fh.write('\n')
    fh.write('assign valid_o = shift_valid;\n')
    fh.write('assign i_output = shift_i[%d:%d];\n' % (input_msb, lsb))
    fh.write('assign q_output = shift_q[%d:%d];\n' % (input_msb, lsb))
    fh.write('\n')
    temp = loop_slice_msb - log_width + 1
    fh.write('assign loop_val_slice = '
             'loop_val[%d:%d];\n' % (loop_slice_msb, temp))
    fh.write('\n')
    fh.write('process(clk)\n')
    fh.write('begin\n')
    fh.write('    reset_n <= !sync_reset;\n')
    fh.write('    reset_n_d1 <= reset_n;\n')
    fh.write('    areset <= reset_n & reset_n_d1;\n')
    fh.write('\n')
    fh.write('    sync_reset_shift_d1 <= sync_reset;\n')
    fh.write('    sync_reset_shift_d2 <= sync_reset_shift_d1;\n')
    fh.write('    sync_reset_shift_d3 <= sync_reset_shift_d2;\n')
    fh.write('\n')
    fh.write('    sync_reset_log_d1 <= sync_reset;\n')
    fh.write('    sync_reset_log_d2 <= sync_reset_log_d1;\n')
    fh.write('    sync_reset_log_d3 <= sync_reset_log_d2;\n')
    fh.write('\n')
    fh.write('    sync_resetExp_d1 <= sync_reset;\n')
    fh.write('    sync_resetExp_d2 <= sync_resetExp_d1;\n')
    fh.write('    sync_resetExp_d3 <= sync_resetExp_d2;\n')
    fh.write('\n')
    fh.write('    sync_resetLoop_d1 <= sync_reset;\n')
    fh.write('    sync_resetLoop_d2 <= sync_resetLoop_d1;\n')
    fh.write('    sync_resetLoop_d3 <= sync_resetLoop_d2;\n')
    fh.write('end if;\n')
    fh.write('\n')
    fh.write('process(clk, sync_reset)\n')
    fh.write('begin\n')
    mult_fac_reset = exp_table.bin[0][2:]
    fh.write('    if (sync_reset = \'1\') then\n')
    fh.write('        log_valid_d <= 0;\n')
    fh.write('        shift_val_r <= 0;\n')
    fh.write('        mult_factor_r <= %d\'b%s;\n'
             % (exp_table.word_length, mult_fac_reset))
    fh.write('        loop_val_slice_d <= 0;\n')
    fh.write('        loop_overflow <= \'0\';\n')
    fh.write('        CVal <= 0;\n')
    fh.write('        log_ref_d <= 0;\n')
    fh.write('        log_value_d <= 0;\n')
    fh.write('        k_i_d <= 0;\n')
    fh.write('    else\n')
    fh.write('        log_valid_d <= {log_valid_d[LOOP_INT_LATENCY-1:0],'
             'log_valid};\n')
    fh.write('        shift_val_r <= next_shift_val_r;\n')
    fh.write('        mult_factor_r <= next_mult_factor_r;\n')
    fh.write('        loop_val_slice_d <= next_loop_val_slice_d;\n')
    fh.write('        loop_overflow <= next_loop_overflow;\n')
    fh.write('        CVal <= next_CVal;\n')
    fh.write('        log_ref_d <= next_log_ref_d;\n')
    fh.write('        log_value_d <= next_log_value_d;\n')
    fh.write('        k_i_d <= next_k_i_d;\n')
    fh.write('    end if;\n')
    fh.write('end if;\n')
    fh.write('\n')
    fh.write('always @*\n')
    fh.write('begin\n')
    fh.write('    next_shift_val_r = shift_val_r;\n')
    fh.write('    next_mult_factor_r = mult_factor_r;\n')
    fh.write('    next_loop_val_slice_d = loop_val_slice_d;\n')
    fh.write('    next_loop_overflow = \'0\';\n')
    fh.write('    next_CVal = CVal;\n')
    fh.write('    if (ExpStb = \'1\') then\n')
    fh.write('        next_shift_val_r = shift_val;\n')
    fh.write('        next_mult_factor_r = lin_val;\n')
    fh.write('    end if;\n')
    fh.write('    --mux Process Latency 1.\n')
    fh.write('    if (loop_val[47] != loop_val_slice[%d]) then\n' % log_msb)
    fh.write('        next_loop_overflow = \'1\';\n')
    fh.write('        if (loop_val[47] = \'1\') then\n')
    str_val = '1' + (log_width - 1) * '0'
    str_val2 = '0' + (log_width - 1) * '1'
    fh.write('            next_loop_val_slice_d = %d\'b%s;\n'
             % (log_width, str_val))
    fh.write('            next_CVal =  {{{%d{loop_val[47]}}, ''{%d{\'0\'}}}};\n'.format(47 - loop_slice_bits,
                                                                                        loop_slice_bits - 1))
    fh.write('        else\n')
    fh.write('            next_loop_val_slice_d = %d\'b%s;\n'.format(log_width, str_val2))
    fh.write('            next_CVal =  {{{%d{loop_val[47]}}, {%d{\'1\'}}}};\n'.format(47 - loop_slice_bits + 1,
                                                                                      loop_slice_bits - 1))
    fh.write('        end if;\n')
    fh.write('    else\n')
    fh.write('        next_loop_val_slice_d = loop_val_slice;\n')
    fh.write('    end if;\n')
    fh.write('\n')
    fh.write('    if (log_valid = \'1\') then\n')
    fh.write('        if (overflow_fil = \'1\') then\n')
    fh.write('            next_k_i_d = k_i_overflow;\n')
    fh.write('            next_log_ref_d = 0;\n')
    temp = fp_utils.fi(mag_s_table.range.max,
                       numeric_type=mag_s_table.numeric_type)
    fh.write('            next_log_value_d = 18\'b%s;\n' % temp.bin[2:])
    fh.write('        else\n')
    fh.write('            next_k_i_d = k_i;\n')
    fh.write('            next_log_ref_d = log_ref_i;\n')
    fh.write('            next_log_value_d = log_value;\n')
    fh.write('        end if;\n')
    fh.write('    else\n')
    fh.write('        next_log_ref_d = 0;\n')
    fh.write('        next_log_value_d = 0;\n')
    fh.write('        next_k_i_d = 0;\n')
    fh.write('    end if;\n')
    fh.write('\n')
    fh.write('end if;\n')
    fh.write('\n')
    fh.write('%s Shifter\n' % shifter_name)
    fh.write('(\n')
    fh.write('  .sync_reset(sync_reset_shift_d3),\n')
    fh.write('  .clk(clk),\n')
    fh.write('\n')
    fh.write('  .valid_i(valid_i), \n')
    fh.write('  .mult_factor_i(mult_factor_r),\n')
    fh.write('  .shift_factor_i(shift_val_r),\n')
    fh.write('\n')
    fh.write('  .i_input(i_input),\n')
    fh.write('  .q_input(q_input),\n')
    fh.write('\n')
    fh.write('  .valid_o(shift_valid),\n')
    fh.write('  .i_output(shift_i),\n')
    fh.write('  .q_output(shift_q),\n')
    fh.write('  .overflow_o(overflow)\n')
    fh.write('\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('%s Cordic (\n' % cordic_name)
    fh.write('  .aclk(clk), -- input aclk\n')
    fh.write('  .s_axis_cartesian_tvalid(shift_valid), \n')
    fh.write('  .s_axis_cartesian_tdata(CordicInput), \n')
    fh.write('  .s_axis_cartesian_tuser(overflow),\n')
    fh.write('  .m_axis_dout_tvalid(CordicStb), \n')
    fh.write('  .m_axis_dout_tuser(overflow_cordic),\n')
    fh.write('  .m_axis_dout_tdata(CordicOutput)\n')
    fh.write(');\n')
    fh.write('\n')

    frac_8 = input_width % 8
    if (frac_8 != 0):
        pad = 8 - frac_8
        temp = str(pad) + '\'b' + pad * '0'
        str_val = '{%s,CordicOutput[%d:0]}'.format(temp, input_msb)
    else:
        str_val = 'CordicOutput[%d:0]' % input_msb
    fh.write('%s Fil (\n' % filter_name)
    fh.write('  .aclk(clk),\n')
    fh.write('  .s_axis_data_tvalid(CordicStb),\n')
    fh.write('  .s_axis_data_tready(),\n')
    fh.write('  .s_axis_data_tuser(overflow_cordic),\n')
    fh.write('  .s_axis_data_tdata(%s),\n' % str_val)
    fh.write('  .m_axis_data_tvalid(fil_valid),\n')
    fh.write('  .m_axis_data_tuser(overflow_fil),\n')
    fh.write('  .m_axis_data_tdata(fil_data)\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('%s LogConv\n' % log_name)
    fh.write('(\n')
    fh.write('  .sync_reset(sync_reset_log_d3), -- reset\n')
    fh.write('  .clk(clk), -- clock\n')
    fh.write('\n')
    fh.write('  .valid_i(fil_valid),\n')
    fh.write('  .Mag_i(Mag),\n')
    fh.write('\n')
    fh.write('  .valid_o(log_valid),\n')
    fh.write('  .log_value_o(log_value)\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('-- Latency = 4\n')
    fh.write('%s LoopInt (\n' % int_name)
    fh.write('  .clk(clk), \n')
    fh.write('  .a(log_value_d),\n')
    fh.write('  .sel(loop_overflow), \n')
    fh.write('  .sclr(sync_resetLoop_d3),\n')
    fh.write('  .b(k_i_d),\n')
    fh.write('  .c(CVal),\n')
    fh.write('  .d(log_ref_d),\n')
    fh.write('  .p(loop_val) \n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('\n')
    fh.write('%s ExpConv (\n' % exp_name)
    fh.write('  .sync_reset(sync_resetExp_d3), -- reset\n')
    fh.write('  .clk(clk), -- clock\n')
    fh.write('\n')
    fh.write('  .valid_i(log_valid_d[4]),\n')
    fh.write('  .log_i(loop_val_slice_d),\n')
    fh.write('\n')
    fh.write('  .valid_o(ExpStb),\n')
    fh.write('  .lin_val_o(lin_val),\n')
    fh.write('  .shift_val_o(shift_val)\n')
    fh.write('\n')
    fh.write(');\n')
    fh.write('\n')
    fh.write('endmodule\n')
    fh.close()

    gen_exp_conv(exp_file,
                 project_file,
                 exp_combined_fi,
                 corr_fac_fi,
                 input_width=mag_s_table.word_length,
                 table_bits=exp_width)

    gen_shifter(shifter_file,
                project_file,
                input_width=input_width,
                shift_bits=shift_bits,
                gain_width=exp_width,
                device=device,
                family=family,
                package=package,
                speed_grade=speed_grade)

#    x_utils.xco_cordic(cordic_file,
#                       project_file,
#                       input_width,
#                       input_width,
#                       'Translate',
#                       iterations,
#                       precision,
#                       1,
#                       'Embedded_Multiplier',
#                       device,
#                       family,
#                       package,
#                       speed_grade)
#
#    x_utils.xco_fir(filter_file,
#                    project_file,
#                    coef_vals_fi,
#                    device=device,
#                    input_width=fil_in_width,
#                    output_width=fil_out_width,
#                    sample_period=sample_period,
#                    tuser_width=1,
#                    data_sign='Unsigned',
#                    family=family,
#                    package=package,
#                    speed_grade=speed_grade)
#
#    x_utils.xco_dsp48_macro(int_file, project_file,
#                            a_width=log_width,
#                            b_width=18,
#                            device=device,
#                            family=family,
#                            package=package,
#                            speed_grade=speed_grade,
#                            c_width=48,
#                            d_width=log_width,
#                            areg_1=True,
#                            areg_2=True,
#                            areg_3=True,
#                            areg_4=True,
#                            breg_1=True,
#                            breg_2=True,
#                            breg_3=True,
#                            breg_4=True,
#                            dreg_1=True,
#                            sclr=True,
#                            instruction1='(D-A)*B+P',
#                            instruction2='C')

    gen_log_conv(log_file, project_file, log_combined_fi, 0)

    return (agc_log_ref_fi, mag_table, mag_s_table, ki_fi)

def gen_cordic(path, qvec_in=(16, 15), output_width=16, num_iters=6, function='vector', prefix='', tuser_width=0, tlast=False):
    """
        Generates logic for Cordic core. User specifies bitwidths and number of iterations

        ==========
        Parameters
        ==========

            qvec_in : Input quantization vector.
            output_width : I and Q output widths.
            num_iters : number of cordic rotations (includes coarse correction)
            function : String specifying Cordic function approximation.
            prefix : Naming prefix.

    """
    assert(path is not None), 'User must specify Path'
    path = ret_valid_path(path)
    input_width = qvec_in[0]

    module_name = '{}cordic_{}_{}iw_{}iters'.format(prefix, function, input_width, num_iters)
    file_name = name_help(module_name, path)
    module_name = ret_module_name(file_name)

    input_msb = 2 * qvec_in[0] - 1
    frac_bits = qvec_in[0] - qvec_in[1]
    output_msb = 2 * output_width - 1
    corr_qvec = (25, 24)
    angle_qvec = (output_width, output_width - frac_bits)
    int_msb = output_width - 1
    tot_latency = num_iters + 5 + 1  # 5 dsp, 1 for input mux and 1 for each iteration.
    fifo_addr_width = ret_addr_width(tot_latency * 2)
    almost_full_thresh = 1 << (fifo_addr_width - 1)

    with open(file_name, "w") as fh:

        fh.write('--***************************************************************************--\n')
        fh.write('--\n')
        fh.write('-- Author      : PJV\n')
        fh.write('-- File        : {}\n'.format(module_name))
        fh.write('-- Description : Cordic ({} operation).\n'.format(function))
        fh.write('--\n')
        print_header(fh)
        fh.write('--\n')
        fh.write('--***************************************************************************--\n')
        fh.write('\n')
        print_libraries(fh)
        fh.write('package {}_cmp is\n'.format(module_name))
        fh.write('    component {}\n'.format(module_name))
        if tuser_width:
            fh.write('        generic\n')
            fh.write('        (\n')
            fh.write('            TUSER_WIDTH : integer := 8\n')
            fh.write('        );\n')
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        fh.write('            sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('            s_axis_tvalid : in std_logic;\n')
        fh.write('            s_axis_tdata : in std_logic_vector({} downto 0);\n'.format(input_msb))
        if tuser_width:
            fh.write('            s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('            s_axis_tlast : in std_logic;\n')
        fh.write('            s_axis_tready : out std_logic;\n')
        fh.write('\n')
        fh.write('            m_axis_tvalid : out std_logic;\n')
        if tuser_width:
            fh.write('            m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('            m_axis_tlast : out std_logic;\n')
        fh.write('            m_axis_tdata : out std_logic_vector({} downto 0);  -- Magnitude and Phase vectors\n'.format(output_msb))
        fh.write('            m_axis_tready : in std_logic\n')
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        fh.write('\n')
        funcs = ['A*B', '-A*B']
        (dsp_file, dsp_name) = gen_dsp48E1(path, module_name, opcode=funcs, areg=2, breg=3, use_ce=False)
        print(dsp_name)
        (_, fifo_name) = gen_axi_fifo(path, tuser_width=tuser_width, tlast=tlast, almost_full=True, almost_empty=False,
                                      count=False, max_delay=0, ram_style='distributed', prefix='')
        print(fifo_name)

        print_libraries(fh)
        fh.write('library work;\n')
        fh.write('use work.{}_cmp.all;\n'.format(dsp_name))
        fh.write('use work.{}_cmp.all;\n'.format(fifo_name))
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        if tuser_width:
            fh.write('    generic\n')
            fh.write('    (\n')
            fh.write('        TUSER_WIDTH : integer := 8\n')
            fh.write('    );\n')
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        fh.write('        sync_reset : in std_logic;\n')
        fh.write('\n')
        fh.write('        s_axis_tvalid : in std_logic;\n')
        fh.write('        s_axis_tdata : in std_logic_vector({} downto 0);\n'.format(input_msb))
        if tuser_width:
            fh.write('        s_axis_tuser : in std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('        s_axis_tlast : in std_logic;\n')
        fh.write('        s_axis_tready : out std_logic;\n')
        fh.write('\n')
        fh.write('        m_axis_tvalid : out std_logic;\n')
        if tuser_width:
            fh.write('        m_axis_tuser : out std_logic_vector(TUSER_WIDTH-1 downto 0);\n')
        if tlast:
            fh.write('        m_axis_tlast : out std_logic;\n')
        fh.write('        m_axis_tdata : out std_logic_vector({} downto 0);  -- Magnitude and Phase vectors\n'.format(output_msb))
        fh.write('        m_axis_tready : in std_logic\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n\n'.format(module_name))
        if tuser_width:
            fh.write('constant TUSER_MSB : integer := TUSER_WIDTH - 1;\n')
            for i in range(tot_latency):
                fh.write('signal tuser_reg{} : std_logic_vector(TUSER_MSB downto 0);\n'.format(i))
        if tlast:
            fh.write('signal tlast_d : std_logic_vector({} downto 0);\n'.format(tot_latency - 1))
        for ii in range(1, num_iters):
            fh.write('signal x_{}, next_x_{} : signed({} downto 0);\n'.format(ii, ii, int_msb))
        for ii in range(1, num_iters):
            fh.write('signal y_{}, next_y_{} : signed({} downto 0);\n'.format(ii, ii, int_msb))
        for ii in range(1, num_iters):
            fh.write('signal z_{}, next_z_{} : signed({} downto 0);\n'.format(ii, ii, int_msb))
        fh.write('\n')
        fh.write('signal x_0, y_0, z_0 : signed({} downto 0);\n'.format(int_msb))
        fh.write('\n')
        # corr_value = 1. / np.sqrt(1. + 2. ** (-2. * num_iters))
        corr_value = np.prod([1. / np.sqrt(1. + 2. ** (-2. * n_val)) for n_val in range(num_iters)])
        corr_fi = fp_utils.sfi(vec=corr_value, qvec=corr_qvec, signed=1)
        # print(corr_fi.vec)
        fh.write('constant CORD_CORR : std_logic_vector(24 downto 0) := std_logic_vector(to_signed({}, 25));\n'.format(corr_fi.vec[0]))
        fh.write('\n')
        angle_values = [.5, -.5, 0]
        angle_fi = fp_utils.sfi(angle_values, qvec=angle_qvec)
        fh.write('constant PI_OVER2 : signed({} downto 0) := to_signed({}, {});\n'.format(int_msb, angle_fi.vec[0], output_width))
        fh.write('constant NEG_PI_OVER2 : signed({} downto 0) := to_signed({}, {});\n'.format(int_msb, angle_fi.udec[1], output_width))
        fh.write('constant ZERO : signed({} downto 0) := to_signed({}, {});\n'.format(int_msb, angle_fi.vec[2], output_width))
        fh.write('\n')
        fh.write('signal valid_d, next_valid_d : std_logic_vector({} downto 0);\n'.format(tot_latency - 1))
        fh.write('signal angle, next_angle : signed({} downto 0);\n'.format(int_msb))
        fh.write('signal angle_d1, angle_d2, angle_d3, angle_d4, angle_d5 : signed({} downto 0);\n'.format(int_msb))
        fh.write('\n')
        fh.write('signal take_data : std_logic;\n')
        fh.write('signal almost_full : std_logic;\n')
        # fh.write('wire ce;\n')
        # fh.write('reg ce_d0;\n')
        fh.write('\n')
        atan_values = [np.arctan(2. ** -n) for n in range(num_iters - 1)]
        atan_fi = fp_utils.sfi(atan_values, qvec=angle_qvec)
        for ii in range(num_iters - 1):
            fh.write('signal atan_val_{} : signed({} downto 0) := to_signed({}, {});\n'.format(ii, int_msb, atan_fi.vec[ii], output_width))
        fh.write('\n')
        fh.write('signal opcode0, next_opcode0 : std_logic_vector(0 downto 0);\n')
        fh.write('signal opcode1, next_opcode1 : std_logic_vector(0 downto 0);\n')
        fh.write('signal fifo_tdata : std_logic_vector({} downto 0);\n'.format(input_width * 2 - 1))
        fh.write('\n')
        fh.write('signal a_term, next_a_term : std_logic_vector(24 downto 0);\n')
        fh.write('signal b_termx, next_b_termx : std_logic_vector(17 downto 0);\n')
        fh.write('signal b_termy, next_b_termy : std_logic_vector(17 downto 0);\n')
        fh.write('\n')
        fh.write('signal x_corr, y_corr : std_logic_vector(47 downto 0);\n')
        fh.write('\n')
        fh.write('signal i_input, q_input : std_logic_vector(17 downto 0);\n')
        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')
        fh.write('take_data <= s_axis_tvalid and (not almost_full) and (not sync_reset);\n')
        # fh.write('assign send_data = (m_axis_tready & occ_reg[{}]) | (~occ_reg[{}] && occ_reg != 0);\n'.format(tot_latency - 1, tot_latency - 1))  #analysis:ignore
        fh.write('s_axis_tready <= not almost_full;\n')
        # fh.write('assign ce = send_data;\n')
        # fh.write('assign m_axis_tvalid = occ_reg[{}];\n'.format(tot_latency - 1))
        fh.write('fifo_tdata <= std_logic_vector(z_{}) & std_logic_vector(x_{});\n'.format(num_iters - 1, num_iters - 1))
        pad = 18 - input_width
        tup_value0 = (pad, input_width - 1, input_width - 1)
        tup_value1 = (pad, input_msb, input_msb, input_width)
        fh.write('i_input <= std_logic_vector(resize(signed(s_axis_tdata({} downto 0)), 18));\n'.format(input_width - 1))
        fh.write('q_input <= std_logic_vector(resize(signed(s_axis_tdata({} downto {})), 18));\n'.format(input_msb, input_width))
        fh.write('\n')
        mult_frac = corr_qvec[1] + qvec_in[1]
        mult_msb = mult_frac + frac_bits - 1  # - 1 since correction factor is less than 1.
        rindx = mult_msb - output_width + 1
        fh.write('x_0 <= signed(x_corr({} downto {}));\n'.format(mult_msb, rindx))
        fh.write('y_0 <= signed(y_corr({} downto {}));\n'.format(mult_msb, rindx))
        fh.write('z_0 <= angle_d4;\n')
        fh.write('\n')
        fh.write('main_clk_proc:\n')
        fh.write('process(clk, sync_reset)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        if (sync_reset = \'1\') then\n')
        fh.write('            angle <= (others => \'0\');\n')
        fh.write('            opcode0 <= "0";\n')
        fh.write('            opcode1 <= "0";\n')
        fh.write('        else\n')
        fh.write('            angle <= next_angle;\n')
        fh.write('            opcode0 <= next_opcode0;\n')
        fh.write('            opcode1 <= next_opcode1;\n')
        fh.write('        end if;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n\n')
        fh.write('-- Register Logic\n')
        fh.write('register_proc:\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        fh.write('        angle_d1 <= angle;\n')
        fh.write('        angle_d2 <= angle_d1;\n')
        fh.write('        angle_d3 <= angle_d2;\n')
        fh.write('        angle_d4 <= angle_d3;\n')
        fh.write('        angle_d5 <= angle_d4;\n')
        if tuser_width:
            fh.write('        tuser_reg0 <= s_axis_tuser;\n')
            for ii in range(1, tot_latency):
                fh.write('        tuser_reg{} <= tuser_reg{};\n'.format(ii, ii - 1))
        if tlast:
            fh.write('        tlast_d <= tlast_d({} downto 0) & s_axis_tlast;\n'.format(tot_latency - 2))

        fh.write('        valid_d <= valid_d({} downto 0) & take_data;\n'.format(tot_latency - 2))
        fh.write('    end if;\n')
        fh.write('end process;\n\n')
        fh.write('-- latency = 5\n')
        fh.write('x_mac : {}\n'.format(dsp_name))
        fh.write('  port map (\n')
        fh.write('      clk => clk,\n')
        fh.write('      opcode => opcode0,\n')
        fh.write('      a => CORD_CORR,\n')
        fh.write('      b => b_termx,\n')
        fh.write('      p => x_corr\n')
        fh.write('  );\n')
        fh.write('\n')
        fh.write('y_mac : {}\n'.format(dsp_name))
        fh.write('  port map (\n')
        fh.write('      clk => clk,\n')
        fh.write('      opcode => opcode1,\n')
        fh.write('      a => CORD_CORR,\n')
        fh.write('      b => b_termy,\n')
        fh.write('      p => y_corr\n')
        fh.write('  );\n')
        fh.write('\n')
        fh.write('\n')
        fh.write('-- Cordic term logic\n')
        fh.write('term_proc:\n')
        fh.write('process(clk)\n')
        fh.write('begin\n')
        fh.write('    if (rising_edge(clk)) then\n')
        for ii in range(1, num_iters):
            fh.write('        x_{} <= next_x_{};\n'.format(ii, ii))
        fh.write('\n')
        for ii in range(1, num_iters):
            fh.write('        y_{} <= next_y_{};\n'.format(ii, ii))
        fh.write('\n')
        for ii in range(1, num_iters):
            fh.write('        z_{} <= next_z_{};\n'.format(ii, ii))
        fh.write('\n')
        fh.write('        a_term <= next_a_term;\n')
        fh.write('        b_termx <= next_b_termx;\n')
        fh.write('        b_termy <= next_b_termy;\n')
        fh.write('    end if;\n')
        fh.write('end process;\n\n')
        fh.write('async_proc:\n')
        fh.write('process(x_1,\n')
        for ii in range(2, num_iters):
            fh.write('        x_{},\n'.format(ii))
        for ii in range(1, num_iters):
            fh.write('        y_{},\n'.format(ii))
        for ii in range(1, num_iters):
            fh.write('        z_{},\n'.format(ii))
        fh.write('        opcode0,\n')
        fh.write('        opcode1,\n')
        fh.write('        i_input,\n')
        fh.write('        q_input)\n')
        fh.write('begin\n')
        fh.write('\n')
        for ii in range(1, num_iters):
            fh.write('    next_x_{} <= x_{};\n'.format(ii, ii))
        fh.write('\n')
        for ii in range(1, num_iters):
            fh.write('    next_y_{} <= y_{};\n'.format(ii, ii))
        fh.write('\n')
        for ii in range(1, num_iters):
            fh.write('    next_z_{} <= z_{};\n'.format(ii, ii))
        fh.write('\n')
        fh.write('    next_opcode0 <= opcode0;\n')
        fh.write('    next_opcode1 <= opcode1;\n')
        fh.write('\n')
        fh.write('    if (i_input({}) = \'1\' and q_input({}) = \'1\') then\n'.format(int_msb, int_msb))
        fh.write('        next_b_termx <= q_input;\n')
        fh.write('        next_b_termy <= i_input;\n')
        fh.write('        next_opcode0 <= "1";\n')
        fh.write('        next_opcode1 <= "0";\n')
        fh.write('        next_angle <= NEG_PI_OVER2;\n')
        fh.write('\n')
        fh.write('    elsif (i_input({}) = \'1\' and q_input({}) = \'0\') then\n'.format(int_msb, int_msb))
        fh.write('        next_b_termx <= q_input;\n')
        fh.write('        next_b_termy <= i_input;\n')
        fh.write('        next_opcode0 <= "0";\n')
        fh.write('        next_opcode1 <= "1";\n')
        fh.write('        next_angle <= PI_OVER2;\n')
        fh.write('    else\n')
        fh.write('        next_b_termx <= i_input;\n')
        fh.write('        next_b_termy <= q_input;\n')
        fh.write('        next_opcode0 <= "0";\n')
        fh.write('        next_opcode1 <= "0";\n')
        fh.write('        next_angle <= ZERO;\n')
        fh.write('    end if;\n')
        fh.write('\n')
        # fh.write('    if (send_data == \'1\') begin\n')
        for ii in range(1, num_iters):
            fh.write('    if (y_{}({}) = \'1\') then\n'.format(ii - 1, int_msb))
            fh.write('        next_x_{} <= x_{} - shift_right(y_{}, {});\n'.format(ii, ii - 1, ii - 1, ii-1))
            fh.write('        next_y_{} <= y_{} + shift_right(x_{}, {});\n'.format(ii, ii - 1, ii - 1, ii-1))
            fh.write('        next_z_{} <= z_{} - atan_val_{};\n'.format(ii, ii - 1, ii - 1))
            fh.write('    else\n')
            fh.write('        next_x_{} <= x_{} + shift_right(y_{}, {});\n'.format(ii, ii - 1, ii - 1, ii-1))
            fh.write('        next_y_{} <= y_{} - shift_right(x_{}, {});\n'.format(ii, ii - 1, ii - 1, ii-1))
            fh.write('        next_z_{} <= z_{} + atan_val_{};\n'.format(ii, ii - 1, ii - 1))
            fh.write('    end if;\n')
        fh.write('\n')
        fh.write('end process;\n\n')
        fh.write('\n')
        # insert axi fifo for interface compliance
        axi_fifo_inst(fh, fifo_name, inst_name='axi_fifo', data_width=input_width*2, af_thresh=almost_full_thresh,
                      addr_width=fifo_addr_width, tuser_width=tuser_width, tlast=tlast, s_tvalid_str='valid_d{}'.format(tot_latency-1),
                      s_tdata_str='fifo_tdata', s_tuser_str='tuser_reg{}'.format(tot_latency-1), s_tlast_str='tlast_d{}'.format(tot_latency-1),
                      s_tready_str='open', almost_full_str='almost_full', m_tvalid_str='m_axis_tvalid', m_tdata_str='m_axis_tdata',
                      m_tuser_str='m_axis_tuser', m_tlast_str='m_axis_tlast', m_tready_str='m_axis_tready')

        fh.write('\n')
        fh.write('end rtl;\n')

    return module_name

def test_run():
    num_corrs = 20
    gen_pipe_mux(20 * num_corrs, 20, './', one_hot=True)
    input_width = 42
    gen_pipe_logic(input_width)

# if __name__ == "__main__":
#     test_run()
