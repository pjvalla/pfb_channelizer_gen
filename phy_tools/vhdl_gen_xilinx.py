# -*- coding: utf-8 -*-
"""
Created on Tue October 10 17:01:28 2017

@author: phil
"""
import os  
from phy_tools.gen_utils import ret_module_name
import numpy as np
from phy_tools.dsp_opts import opcode_opts
import copy
import ipdb
import re
from phy_tools import fp_utils
from phy_tools.vgen_xilinx import comp_opmodes, ret_opcode_params, use_dict
from phy_tools.gen_utils import print_intro, print_exit

from subprocess import check_output, CalledProcessError, DEVNULL
try:
    __version__ = check_output('git log -1 --pretty=format:%cd --date=format:%Y.%m.%d'.split(), stderr=DEVNULL).decode()
except CalledProcessError:
    from datetime import date
    today = date.today()
    __version__ = today.strftime("%Y.%m.%d")



def print_libraries(fh):
    fh.write('library ieee;\n')
    fh.write('use ieee.std_logic_1164.all;\n')
    fh.write('use ieee.numeric_std.all;\n') #analysis:ignore
    fh.write('library UNISIM;\n')
    fh.write('use UNISIM.Vcomponents.all;\n')

def gen_regs(fh, prefix='a_d', cnt=1, sp='', msb=24, str_val=None):
    for jj in range(cnt):
        if str_val is not None and jj == 0:
            fh.write('{}signal {} : std_logic_vector({} downto 0);\n'.format(sp, str_val, msb))
        else:
            fh.write('{}signal {}{} : std_logic_vector({} downto 0);\n'.format(sp, prefix, jj, msb))

def logic_rst(fh, prefix='a_d', cnt=1, sp=''):
    for jj in range(cnt):
        fh.write('{}{}{} <= (others => \'0\');\n'.format(sp, prefix, jj))

def logic_gate(fh, prefix='a_d', str_val='a', cnt=1, sp=''):
    for jj in range(cnt):
        rside = str_val if (jj == 0) else '{}{}'.format(prefix, jj - 1)
        fh.write('{}{}{} <= {};\n'.format(sp, prefix, jj, rside))


def gen_dsp48E1(path, name, opcode='A*B', a_width=25, b_width=18, c_width=48, d_width=25,
                areg=1, breg=1, creg=1, dreg=1, mreg=1, preg=1, concatreg=1, carryreg=1,
                use_acout=False, use_bcout=False, use_pcout=False, use_ce=False, use_reset=False, rnd=False,
                p_msb=40, p_lsb=18, a_signed=True, b_signed=True):

    def input_port(fh, opcode, sp=''):
        # op_strs = re.findall(r"[\w']+", opcode)
        # for str_val in op_strs
        if use_aport:
            fh.write('    {}a : in std_logic_vector({} downto 0);\n'.format(sp, a_msb))
        if use_acin_port:
            fh.write('    {}acin : in std_logic_vector(29 downto 0);\n'.format(sp))
        if use_bport:
            fh.write('    {}b : in std_logic_vector({} downto 0);\n'.format(sp, b_msb))
        if use_bcin_port:
            fh.write('    {}bcin : in std_logic_vector(29 downto 0);\n'.format(sp))
        if use_concat:
            fh.write('    {}concat : in std_logic_vector(47 downto 0);\n'.format(sp))
        if use_pcin_port:
            fh.write('    {}pcin : in std_logic_vector(47 downto 0);\n'.format(sp))
        if use_carryin:
            fh.write('    {}carryin : in std_logic;\n'.format(sp))
        if use_carrycascin:
            fh.write('    {}carrycascin : in std_logic;\n'.format(sp))
        if use_dport:
            fh.write('    {}d : in std_logic_vector({} downto 0);\n'.format(sp, d_msb))
        if use_cport:
            fh.write('    {}c : in std_logic_vector({} downto 0);\n'.format(sp, c_msb))
        if multi_opcode:
            fh.write('    {}opcode : in std_logic_vector({} downto 0);\n'.format(sp, opcode_bits - 1))

    def output_port(fh, sp=''):
        if use_acout:
            fh.write('    {}acout : out std_logic_vector(29 downto 0);\n'.format(sp))
        if use_bcout:
            fh.write('    {}bcout : out std_logic_vector(29 downto 0);\n'.format(sp))
        if use_pcout:
            fh.write('    {}pcout : out std_logic_vector(47 downto 0);\n'.format(sp))

    assert(path is not None), 'User must specify directory'
    file_name = 'dsp48_{}.vhd'.format(name)
    file_name = os.path.join(path, file_name)
    module_name = ret_module_name(file_name)
    # parse opcode
    if type(opcode) is list:
        opcodes = copy.copy(opcode)
    else:
        opcodes = [opcode]

    concat_list = []
    two_pports_list = []
    mult_list = []
    aport_list = []
    bport_list = []
    cport_list = []
    dport_list = []
    acin_list = []
    bcin_list = []
    pcin_list = []
    carryin_list = []
    carrycascin_list = []

    for opcode in opcodes:
        temp_dict = ret_opcode_params(opcode)
        concat_list.append(temp_dict['use_concat'])
        two_pports_list.append(temp_dict['use_2pports'])
        mult_list.append(temp_dict['use_mult'])
        aport_list.append(temp_dict['use_aport'])
        bport_list.append(temp_dict['use_bport'])
        cport_list.append(temp_dict['use_cport'])
        dport_list.append(temp_dict['use_dport'])
        acin_list.append(temp_dict['use_acin'])
        bcin_list.append(temp_dict['use_bcin'])
        pcin_list.append(temp_dict['use_pcin'])
        carryin_list.append(temp_dict['use_carryin'])
        carrycascin_list.append(temp_dict['use_carrycascin'])

    use_mult = np.any(mult_list)
    use_2pports = np.any(two_pports_list)
    use_concat = np.any(concat_list)
    use_aport = np.any(aport_list)
    use_bport = np.any(bport_list)
    use_cport = np.any(cport_list)
    use_dport = np.any(dport_list)
    use_acin_port = np.any(acin_list)
    use_bcin_port = np.any(bcin_list)
    use_pcin_port = np.any(pcin_list)
    use_carryin = np.any(carryin_list)
    use_carrycascin = np.any(carrycascin_list)
    use_preadd = use_aport and use_dport

    pcin = 'pcin' if use_pcin_port else '(others => \'0\')'
    a_val = 'a_s' if (use_aport or use_concat) else '(others => \'0\')'
    b_val = 'b_s' if (use_bport or use_concat) else '\"000000000000000001\"'
    c_val = 'c_s' if use_cport else '(others => \'0\')'
    d_val = 'd_s' if use_dport else '(others => \'0\')'
    acin = 'acin' if use_acin_port else '(others => \'0\')'
    bcin = 'bcin' if use_bcin_port else '(others => \'0\')'
    carryin = 'carryin' if use_carryin else '\'0\''
    carrycascin = 'carrycascin' if use_carrycascin else '\'0\''

    multi_opcode = (len(opcodes) > 1)
    opcode_bits = int(np.ceil(np.log2(len(opcodes))))
    op_strs = []
    for opcode in opcodes:
        opcode = opcode.lower()
        op_strs.append(re.findall(r"[\w']+", opcode))
    # op_strs = op_code.split(" ") gives a list of strings:
    a_source = 'DIRECT'
    b_source = 'DIRECT'
    areg_logic = 0
    breg_logic = 0
    creg_logic = 0
    dreg_logic = 0
    concatreg_logic = 0
    carryreg_logic = 0

    opmode_reg = 0
    inmode_reg = 0
    alumode_reg = 0
    carryin_sel_reg = 0

    mult_delay = np.max((areg, breg)) + mreg
    opmode_logic = 0
    opmode_delay = 0
    alumode_delay = 0
    inmode_delay = areg
    if use_concat:
        inmode_delay = np.max((inmode_delay, concatreg))
    if use_cport:
        inmode_delay = np.max((inmode_delay, creg - mreg))

    inmode_logic = 0
    alumode_logic = 0
    carryin_sel_delay = 0
    carryin_sel_logic = 0
    if multi_opcode:
        opmode_delay = np.max((mult_delay, creg))
        opmode_reg = np.min((opmode_delay, 1))
        opmode_logic = opmode_delay - opmode_reg
        inmode_reg = 1
        inmode_logic = inmode_delay - inmode_reg
        alumode_delay = mult_delay
        alumode_reg = 1
        alumode_logic = alumode_delay - alumode_reg
        carryin_sel_delay = mult_delay
        carryin_sel_reg = 1
        carryin_sel_logic = carryin_sel_delay - carryin_sel_reg

    infer_logic = False
    input_regs = np.max([(areg - 2)*use_aport, (breg - 2)*use_bport, (creg - 1)*use_cport, (dreg - 1)*use_dport, concatreg - 2])
    if input_regs + alumode_logic + inmode_logic + opmode_logic > 0:
        infer_logic = True

    if use_concat:
        if concatreg > 2:
            breg_logic = concatreg - 2
            areg_logic = concatreg - 2
            areg = 2
            breg = 2
        else:
            areg = concatreg
            breg = concatreg
    else:
        areg_logic = np.max((areg - 2, 0))
        breg_logic = np.max((breg - 2, 0))

    if areg_logic > 0 and (use_aport or use_concat):
        a_val = 'a_d{}'.format(areg_logic - 1)
        areg = 2
    if breg_logic > 0 and (use_bport or use_concat):
        b_val = 'b_d{}'.format(breg_logic - 1)
        breg = 2
    if not use_cport:
        creg = 0
    else:
        if creg > 1 and use_cport:
            creg_logic = creg - 1
            creg = 1
            c_val = 'c_d{}'.format(creg_logic - 1)
    if not use_dport:
        dreg = 0
    else:
        if dreg > 1:
            dreg_logic = dreg - 1
            dreg = 1
            d_val = 'd_d{}'.format(dreg_logic - 1)
    if carryreg > 1:
        carryreg_logic = carryreg - 1
        carryreg = 1

    if use_concat:
        if concatreg > 2:
            breg_logic = concatreg - 2
            areg_logic = concatreg - 2
            areg = 2
            breg = 2
        else:
            areg = concatreg
            breg = concatreg

    a_msb = a_width - 1
    b_msb = b_width - 1
    c_msb = c_width - 1
    d_msb = d_width - 1

    ce = '\'1\''
    if use_ce is True:
        ce = 'ce'

    if areg >= 1:
        cea1 = ce
    else:
        cea1 = '\'0\''

    if areg == 2:
        cea2 = ce
    else:
        cea2 = '\'0\''

    if breg >= 1:
        ceb1 = ce
    else:
        ceb1 = '\'0\''

    if breg == 2:
        ceb2 = ce
    else:
        ceb2 = '\'0\''

    sync_reset = '\'0\''
    if use_reset:
        sync_reset = 'sync_reset'

    a = '(others => \'0\')'
    b = '\"000000000000000001\"'   #'(others => \'0\')'
    c = '(others => \'0\')'
    d = '(others => \'0\')'

    dport_str = 'FALSE'
    if use_dport:
        dport_str = 'TRUE'


    if rnd:
        p_width = p_msb - p_lsb + 1
    else:
        p_width = 48

    if rnd:
        assert(use_cport is not True), 'User cannot use c-port when rounding'
        c_constant = 2**(p_lsb - 1) - 1
        if use_aport:
            carryin = 'a(0)'
        elif use_bport:
            carryin = 'b(0)'
        else:
            # make this a registered value.
            carryin = 'p_d'

    for str_val in op_strs:
        if str_val == 'acin':
            a_source = 'CASCADE'
        if str_val == 'bcin':
            b_source = 'CASCADE'

    opmodes = []
    inmodes = []
    alumodes = []
    carryin_sels = []
    for opcode in opcodes:
        temp0, temp1, temp2, temp3 = comp_opmodes(opcode, areg, breg, rnd)
        # print(opcode, temp0, temp1, temp2, temp3)
        opmodes.append(temp0)
        inmodes.append(temp1)
        alumodes.append(temp2)
        carryin_sels.append(temp3)

    if opmode_logic:
        opmode_str = 'opmode_d{}'.format(opmode_logic - 1)
    else:
        if multi_opcode:
            opmode_str = 'opmode'
        else:
            str_val = fp_utils.dec_to_ubin(opmodes[0], 7)
            opmode_str = '\"{}\"'.format(str_val)

    if inmode_logic:
        inmode_str = 'inmode_d{}'.format(inmode_delay - 2)
    else:
        if multi_opcode:
            inmode_str = 'inmode'
        else:
            str_val = fp_utils.dec_to_ubin(inmodes[0], 5)
            inmode_str = '\"{}\"'.format(str_val)

    if alumode_logic:
        alumode_str = 'alumode_d{}'.format(alumode_delay - 2)
    else:
        if multi_opcode:
            alumode_str = 'alumode'
        else:
            str_val = fp_utils.dec_to_ubin(alumodes[0], 4)
            alumode_str = '\"{}\"'.format(str_val)

    if carryin_sel_logic:
        carryin_sel_str = 'carryin_sel_d{}'.format(carryin_sel_logic - 1)
    else:
        if multi_opcode:
            carryin_sel_str = 'carryin_sel'
        else:
            str_val = fp_utils.dec_to_ubin(carryin_sels[0], 3)
            carryin_sel_str = '\"{}\"'.format(str_val)

    if not use_mult:
        mreg = 0

    with open(file_name, "w") as fh:
        fh.write('\n')
        print_libraries(fh)
        fh.write('\n')
        print_intro(fh, module_name)
        fh.write('        port\n')
        fh.write('        (\n')
        fh.write('            clk : in std_logic;\n')
        if use_reset:
            fh.write('            sync_reset : in std_logic;\n')
        if use_ce:
            fh.write('            ce : in std_logic;\n')
        fh.write('\n')
        input_port(fh, opcode, sp='        ')
        output_port(fh, sp='        ')
        # fh.write('\n')
        if rnd:
            fh.write('            p : out std_logic_vector({} downto 0)\n'.format(p_width - 1))
        else:
            fh.write('            p : out std_logic_vector(47 downto 0)\n')
        fh.write('        );\n')
        print_exit(fh, module_name)
        fh.write('\n')
        fh.write('\n')
        print_libraries(fh)
        # fh.write('library work;\n')
        # fh.write('use UNISIM.vcomponents.all;\n')
        fh.write('\n')
        fh.write('entity {} is\n'.format(module_name))
        fh.write('    port\n')
        fh.write('    (\n')
        fh.write('        clk : in std_logic;\n')
        if use_reset:
            fh.write('        sync_reset : in std_logic;\n')
        if use_ce:
            fh.write('        ce : in std_logic;\n')
        fh.write('\n')

        input_port(fh, opcode, sp='    ')
        output_port(fh, sp='    ')
        # fh.write('\n')
        if rnd:
            fh.write('        p : out std_logic_vector({} downto 0)\n'.format(p_width - 1))
        else:
            fh.write('        p : out std_logic_vector(47 downto 0)\n')
        fh.write('    );\n')
        fh.write('end {};\n'.format(module_name))
        fh.write('\n')
        fh.write('architecture rtl of {} is \n'.format(module_name))
        fh.write('\n')
        if use_aport or use_concat:
            fh.write('signal a_s : std_logic_vector(29 downto 0);\n')
        if use_bport or use_concat:
            fh.write('signal b_s : std_logic_vector(17 downto 0);\n')
        if use_cport:
            fh.write('signal c_s : std_logic_vector(47 downto 0);\n')
        if use_dport:
            fh.write('signal d_s : std_logic_vector(24 downto 0);\n')
        if rnd:
            fh.write('signal p_s : std_logic_vector(47 downto 0) := (others => \'0\');\n')
            fh.write('signal p_d : std_logic := \'0\';\n')
        fh.write('\n')
        if infer_logic:
            gen_regs(fh, prefix='a_d', cnt=areg_logic, sp='', msb=29)
            gen_regs(fh, prefix='b_d', cnt=breg_logic, sp='', msb=17)
            gen_regs(fh, prefix='c_d', cnt=creg_logic, sp='', msb=47)
            gen_regs(fh, prefix='d_d', cnt=dreg_logic, sp='', msb=24)
            gen_regs(fh, prefix='carryreg_d', cnt=carryreg_logic, sp='', msb=0)
            # print("opmode_logic = {}".format(opmode_logic))
            gen_regs(fh, prefix='opmode_d', cnt=opmode_logic, sp='', msb=6)
            gen_regs(fh, prefix='inmode_d', cnt=inmode_logic, sp='', msb=4)
            gen_regs(fh, prefix='alumode_d', cnt=alumode_logic, sp='', msb=3)
            gen_regs(fh, prefix='carryin_sel_d', cnt=carryin_sel_logic, sp='', msb=2)
            if multi_opcode is True:
                gen_regs(fh, prefix='next_alumode', cnt=1, sp='', msb=3, str_val='next_alumode')
                gen_regs(fh, prefix='next_carryin_sel', cnt=1, sp='', msb=2, str_val='next_carryin_sel')
                gen_regs(fh, prefix='next_inmode', cnt=1, sp='', msb=4, str_val='next_inmode')
                gen_regs(fh, prefix='next_opmode', cnt=1, sp='', msb=6, str_val='next_opmode')

        fh.write('\n')
        fh.write('begin\n')
        fh.write('\n')

        if rnd:
            fh.write('p <= p_s({} downto {});\n'.format(p_msb, p_lsb))
        if use_aport or use_concat:
            if use_concat:
                fh.write('a_s <= concat(47 downto 18);\n')
            else:
                if a_signed:
                    fh.write('a_s <= std_logic_vector(resize(signed(a), 30));\n')
                else:
                    fh.write('a_s <= std_logic_vector(resize(unsigned(a), 30));\n')

        if use_bport or use_concat:
            if use_concat:
                fh.write('b_s <= concat(17 downto 0);\n')
            else:
                if b_width < 18:
                    if b_signed:
                        fh.write('b_s <= std_logic_vector(resize(signed(b), 18));\n')
                    else:
                        fh.write('b_s <= std_logic_vector(resize(unsigned(b), 18));\n')

                else:
                    fh.write('b_s <= b;\n')

        if use_dport:
            if d_width < 25:
                fh.write('d_s <= std_logic_vector(resize(signed(d), 25));\n')
            else:
                fh.write('d_s <= d;\n')

        if use_cport:
            if c_width < 48:
                fh.write('c_s <= std_logic_vector(resize(signed(c), 48));\n')
            else:
                fh.write('c_s <= c;\n')

        fh.write('\n')
        extra_tab = ''
        if infer_logic or rnd:
            fh.write('process(clk)\n')
            fh.write('begin\n')
            # old data is presented on the output port
            fh.write('    if (rising_edge(clk)) then\n')
            if use_reset:
                fh.write('	      if (sync_reset = \'1\') then\n')
                logic_rst(fh, prefix='a_d', cnt=areg_logic, sp='\t\t\t')
                logic_rst(fh, prefix='b_d', cnt=breg_logic, sp='\t\t\t')
                logic_rst(fh, prefix='c_d', cnt=creg_logic, sp='\t\t\t')
                logic_rst(fh, prefix='d_d', cnt=dreg_logic, sp='\t\t\t')
                logic_rst(fh, prefix='carryreg_d', cnt=carryreg_logic, sp='\t\t\t')
                logic_rst(fh, prefix='opmode_d', cnt=opmode_logic, sp='\t\t\t')
                logic_rst(fh, prefix='inmode_d', cnt=inmode_logic, sp='\t\t\t')
                logic_rst(fh, prefix='alumode_d', cnt=alumode_logic, sp='\t\t\t')
                logic_rst(fh, prefix='carryin_sel_d', cnt=carryin_sel_logic, sp='\t\t\t')
                logic_rst(fh, prefix='carryin_sel_d', cnt=carryin_sel_logic, sp='\t\t\t')
                if rnd:
                    fh.write('	          p_d <= \'0\';\n')

                fh.write('        else\n')
                extra_tab = '\t'
                if use_ce:
                    extra_tab = '\t\t'
                    fh.write('	          if (ce = \'1\') then\n')
                logic_gate(fh, prefix='a_d', str_val='a_s', cnt=areg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='b_d', str_val='b_s', cnt=breg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='c_d', str_val='c_s', cnt=creg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='d_d', str_val='d_s', cnt=dreg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='carryreg_d', str_val='carryreg', cnt=carryreg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='opmode_d', str_val='next_opmode', cnt=opmode_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                logic_gate(fh, prefix='inmode_d', str_val='next_inmode', cnt=inmode_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                logic_gate(fh, prefix='alumode_d', str_val='next_alumode', cnt=alumode_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                logic_gate(fh, prefix='carryin_sel_d', str_val='next_carryin_sel', cnt=carryin_sel_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                if rnd:
                    fh.write('\t{}p_d <= p_s(0);\n'.format(extra_tab))
                #
                if use_ce:
                    fh.write('            end if;\n')
                fh.write('        end if;\n')
            else:
                extra_tab = '\t'
                if use_ce:
                    extra_tab = '\t\t'
                    fh.write('        if (ce = \'1\') then\n')
                logic_gate(fh, prefix='a_d', str_val='a_s', cnt=areg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='b_d', str_val='b_s', cnt=breg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='c_d', str_val='c_s', cnt=creg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='d_d', str_val='d_s', cnt=dreg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='carryreg_d', str_val='carryreg', cnt=carryreg_logic, sp='\t{}'.format(extra_tab))
                logic_gate(fh, prefix='opmode_d', str_val='next_opmode', cnt=opmode_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                logic_gate(fh, prefix='alumode_d', str_val='next_alumode', cnt=alumode_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                logic_gate(fh, prefix='inmode_d', str_val='next_inmode', cnt=inmode_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                logic_gate(fh, prefix='carryin_sel_d', str_val='next_carryin_sel', cnt=carryin_sel_logic, sp='\t{}'.format(extra_tab))  #analysis:ignore
                if rnd:
                    fh.write('\t{}p_d <= p_s(0);\n'.format(extra_tab))
                if use_ce:
                    fh.write('        end if;\n')
            fh.write('    end if;\n')
            fh.write('end process;\n\n')

        if multi_opcode:
            extra_tab = ''
            fh.write('async_proc:\n')
            fh.write('process(')
            if opmode_logic:
                fh.write('opmode_d0, ')
            if inmode_logic:
                fh.write('inmode_d0, ')
            if carryreg_logic:
                fh.write('carryin_sel_d0, ')
            if use_ce:
                fh.write('ce, ')

            if alumode_logic:
                fh.write('alumode_d0, ')
            fh.write(' opcode')
            fh.write(')\n')
            # fh.write('always @*\n')
            fh.write('begin\n')
            fh.write('    next_opmode <= opmode_d0;\n')
            fh.write('    next_inmode <= inmode_d0;\n')
            fh.write('    next_alumode <= alumode_d0;\n')
            if carryreg_logic:
                fh.write('    next_carryin_sel <= carryin_sel_d0;\n')
            if use_ce:
                fh.write('    if (ce = \'1\') then\n')
                extra_tab = '\t'
            for ii in range(len(opcodes)):
                str_val = fp_utils.dec_to_ubin(ii, opcode_bits)
                opmode_val = fp_utils.dec_to_ubin(opmodes[ii], 7)
                alumode_val = fp_utils.dec_to_ubin(alumodes[ii], 4)
                inmode_val = fp_utils.dec_to_ubin(inmodes[ii], 5)
                carryin_sel_val = fp_utils.dec_to_ubin(carryin_sels[ii], 3)

                if ii == 0:
                    fh.write('{}    if (opcode = \"{}\") then\n'.format(extra_tab, str_val))
                else:
                    fh.write('{}    elsif (opcode = \"{}\") then\n'.format(extra_tab, str_val))
                fh.write('{}        next_opmode <= \"{}\";\n'.format(extra_tab, opmode_val))
                fh.write('{}        next_alumode <= \"{}\";\n'.format(extra_tab, alumode_val))
                fh.write('{}        next_inmode <= \"{}\";\n'.format(extra_tab, inmode_val))
                fh.write('{}        next_carryin_sel <= \"{}\";\n'.format(extra_tab, carryin_sel_val))
            fh.write('{}    else\n'.format(extra_tab))
            opmode_val = fp_utils.dec_to_ubin(opmodes[0], 7)
            alumode_val = fp_utils.dec_to_ubin(alumodes[0], 4)
            inmode_val = fp_utils.dec_to_ubin(inmodes[0], 5)
            carryin_sel_val = fp_utils.dec_to_ubin(carryin_sels[0], 3)
            fh.write('{}        next_opmode <= \"{}\";\n'.format(extra_tab, opmode_val))
            fh.write('{}        next_alumode <= \"{}\";\n'.format(extra_tab, alumode_val))
            fh.write('{}        next_inmode <= \"{}\";\n'.format(extra_tab, inmode_val))
            fh.write('{}        next_carryin_sel <= \"{}\";\n'.format(extra_tab, carryin_sel_val))
            fh.write('{}    end if;\n'.format(extra_tab))
            if use_ce:
                fh.write('    end if;\n')
            fh.write('end process;\n\n')
        fh.write('DSP48E1_inst : DSP48E1\n')
        fh.write('generic map (\n')
        areg_final = areg
        fh.write('    -- Feature Control Attributes: Data Path Selection\n')
        fh.write('    A_INPUT => "{}", -- Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)\n'.format(a_source))  #analysis:ignore
        fh.write('    B_INPUT => "{}", -- Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)\n'.format(b_source))  #analysis:ignore
        fh.write('    USE_DPORT => {}, -- Select D port usage (TRUE or FALSE)\n'.format(dport_str))
        if use_concat and use_mult:
            fh.write('    USE_MULT => "DYNAMIC", -- Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")\n')
        elif use_concat:
            fh.write('    USE_MULT => "NONE", -- Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")\n')
        else:
            fh.write('    USE_MULT => "MULTIPLY", -- Select multiplier usage ("MULTIPLY", "DYNAMIC", or "NONE")\n')
        fh.write('    -- Pattern Detector Attributes: Pattern Detection Configuration\n')
        fh.write('    AUTORESET_PATDET => "NO_RESET", -- "NO_RESET", "RESET_MATCH", "RESET_NOT_MATCH"\n')
        fh.write('    MASK => x"3fffffffffff", -- 48-bit mask value for pattern detect (1=ignore)\n')
        fh.write('    PATTERN => x"000000000000", -- 48-bit pattern match for pattern detect\n')
        fh.write('    SEL_MASK => "MASK", -- "C", "MASK", "ROUNDING_MODE1", "ROUNDING_MODE2"\n')
        fh.write('    SEL_PATTERN => "PATTERN", -- Select pattern value ("PATTERN" or "C")\n')
        fh.write('    USE_PATTERN_DETECT => "NO_PATDET", -- Enable pattern detect ("PATDET" or "NO_PATDET")\n')
        fh.write('    -- Register Control Attributes: Pipeline Register Configuration\n')
        fh.write('    ACASCREG => {}, -- Number of pipeline stages between A/ACIN and ACOUT (0, 1 or 2)\n'.format(areg_final))
        fh.write('    ADREG => 1, -- Number of pipeline stages for pre-adder (0 or 1)\n')
        fh.write('    ALUMODEREG => {}, -- Number of pipeline stages for ALUMODE (0 or 1)\n'.format(alumode_reg))
        fh.write('    AREG => {}, -- Number of pipeline stages for A (0, 1 or 2)\n'.format(areg_final))
        fh.write('    BCASCREG => {}, -- Number of pipeline stages between B/BCIN and BCOUT (0, 1 or 2)\n'.format(breg))
        fh.write('    BREG => {}, -- Number of pipeline stages for B (0, 1 or 2)\n'.format(breg))
        fh.write('    CARRYINREG => {}, -- Number of pipeline stages for CARRYIN (0 or 1)\n'.format(carryreg))
        fh.write('    CARRYINSELREG => 1, -- Number of pipeline stages for CARRYINSEL (0 or 1)\n')
        fh.write('    CREG => {}, -- Number of pipeline stages for C (0 or 1)\n'.format(creg))
        fh.write('    DREG => {}, -- Number of pipeline stages for D (0 or 1)\n'.format(dreg))
        fh.write('    INMODEREG => {}, -- Number of pipeline stages for INMODE (0 or 1)\n'.format(inmode_reg))
        fh.write('    MREG => {}, -- Number of multiplier pipeline stages (0 or 1)\n'.format(mreg))
        fh.write('    OPMODEREG => {}, -- Number of pipeline stages for OPMODE (0 or 1)\n'.format(opmode_reg))
        fh.write('    PREG => {}, -- Number of pipeline stages for P (0 or 1)\n'.format(preg))
        fh.write('    USE_SIMD => "ONE48" -- SIMD selection ("ONE48", "TWO24", "FOUR12")\n')
        fh.write(')\n')
        fh.write('port map (\n')
        fh.write('    -- Cascade: 30-bit (each) output: Cascade Ports\n')
        if use_acout:
            fh.write('    ACOUT => acout, -- 30-bit output: A port cascade output\n')
        else:
            fh.write('    ACOUT => open, -- 30-bit output: A port cascade output\n')

        if use_bcout:
            fh.write('    BCOUT => bcout, -- 18-bit output: B port cascade output\n')
        else:
            fh.write('    BCOUT => open, -- 18-bit output: B port cascade output\n')
        fh.write('    CARRYCASCOUT => open, -- 1-bit output: Cascade carry output\n')
        fh.write('    MULTSIGNOUT => open, -- 1-bit output: Multiplier sign cascade output\n')
        if use_pcout:
            fh.write('    PCOUT => pcout, -- 48-bit output: Cascade output\n')
        else:
            fh.write('    PCOUT => open, -- 48-bit output: Cascade output\n')
        fh.write('    -- Control: 1-bit (each) output: Control Inputs/Status Bits\n')
        fh.write('    OVERFLOW => open, -- 1-bit output: Overflow in add/acc output\n')
        fh.write('    PATTERNBDETECT => open, -- 1-bit output: Pattern bar detect output\n')
        fh.write('    PATTERNDETECT => open, -- 1-bit output: Pattern detect output\n')
        fh.write('    UNDERFLOW => open, -- 1-bit output: Underflow in add/acc output\n')
        fh.write('    -- Data: 4-bit (each) output: Data Ports\n')
        fh.write('    CARRYOUT => open, -- 4-bit output: Carry output\n')
        if rnd:
            fh.write('    P => p_s, -- 48-bit output: Primary data output\n')
        else:
            fh.write('    P => p, -- 48-bit output: Primary data output\n')
        fh.write('    -- Cascade: 30-bit (each) input: Cascade Ports\n')
        fh.write('    ACIN => {}, -- 30-bit input: A cascade data input\n'.format(acin))
        fh.write('    BCIN => {}, -- 18-bit input: B cascade input\n'.format(bcin))
        fh.write('    CARRYCASCIN => {}, -- 1-bit input: Cascade carry input\n'.format(carrycascin))
        fh.write('    MULTSIGNIN => \'0\', -- 1-bit input: Multiplier sign input\n')
        fh.write('    PCIN => {}, -- 48-bit input: P cascade input\n'.format(pcin))
        fh.write('    -- Control: 4-bit (each) input: Control Inputs/Status Bits\n')
        fh.write('    ALUMODE => {}, -- 4-bit input: ALU control input\n'.format(alumode_str))
        fh.write('    CARRYINSEL => {}, -- 3-bit input: Carry select input\n'.format(carryin_sel_str))
        if multi_opcode:
            fh.write('    CEINMODE => {}, -- 1-bit input: Clock enable input for INMODEREG\n'.format(ce))
        else:
            fh.write('    CEINMODE => \'1\', -- 1-bit input: Clock enable input for INMODEREG\n')

        fh.write('    CLK => clk, -- 1-bit input: Clock input\n')
        fh.write('    INMODE => {}, -- 5-bit input: INMODE control input\n'.format(inmode_str))
        fh.write('    OPMODE => {}, -- 7-bit input: Operation mode input\n'.format(opmode_str))
        fh.write('    RSTINMODE => {}, -- 1-bit input: Reset input for INMODEREG\n'.format(sync_reset))
        fh.write('    -- Data: 30-bit (each) input: Data Ports\n')
        fh.write('    A => {}, -- 30-bit input: A data input\n'.format(a_val))
        fh.write('    B => {}, -- 18-bit input: B data input\n'.format(b_val))
        if rnd:
            str_val = fp_utils.dec_to_ubin(c_constant, 48)
            fh.write('    C => \"{}\", -- 48-bit input: C data input\n'.format(str_val))
        else:
            # str_val = fp_utils.dec_to_ubin(c_constant, 48)
            fh.write('    C => {}, -- 48-bit input: C data input\n'.format(c_val))
        fh.write('    CARRYIN => {}, -- 1-bit input: Carry input signal\n'.format(carryin))
        fh.write('    D => {}, -- 25-bit input: D data input\n'.format(d_val))
        fh.write('    -- Reset/Clock Enable: 1-bit (each) input: Reset/Clock Enable Inputs\n')
        fh.write('    CEA1 => {}, -- 1-bit input: Clock enable input for 1st stage AREG\n'.format(cea1))
        fh.write('    CEA2 => {}, -- 1-bit input: Clock enable input for 2nd stage AREG\n'.format(cea2))
        fh.write('    CEAD => {}, -- 1-bit input: Clock enable input for ADREG\n'.format(ce))
        if multi_opcode:
            fh.write('    CEALUMODE => {}, -- 1-bit input: Clock enable input for ALUMODERE\n'.format(ce))
        else:
            fh.write('    CEALUMODE => \'1\', -- 1-bit input: Clock enable input for ALUMODERE\n')
        fh.write('    CEB1 => {}, -- 1-bit input: Clock enable input for 1st stage BREG\n'.format(ceb1))
        fh.write('    CEB2 => {}, -- 1-bit input: Clock enable input for 2nd stage BREG\n'.format(ceb2))
        fh.write('    CEC => {}, -- 1-bit input: Clock enable input for CREG\n'.format(ce))
        fh.write('    CECARRYIN => {}, -- 1-bit input: Clock enable input for CARRYINREG\n'.format(ce))
        if multi_opcode:
            fh.write('    CECTRL => {}, -- 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG\n'.format(ce))
        else:
            fh.write('    CECTRL => \'1\', -- 1-bit input: Clock enable input for OPMODEREG and CARRYINSELREG\n')
        fh.write('    CED => {}, -- 1-bit input: Clock enable input for DREG\n'.format(ce))
        fh.write('    CEM => {}, -- 1-bit input: Clock enable input for MREG\n'.format(ce))
        fh.write('    CEP => {}, -- 1-bit input: Clock enable input for PREG\n'.format(ce))
        fh.write('    RSTA => {}, -- 1-bit input: Reset input for AREG\n'.format(sync_reset))
        fh.write('    RSTALLCARRYIN => {}, -- 1-bit input: Reset input for CARRYINREG\n'.format(sync_reset))
        fh.write('    RSTALUMODE => {}, -- 1-bit input: Reset input for ALUMODEREG\n'.format(sync_reset))
        fh.write('    RSTB => {}, -- 1-bit input: Reset input for BREG\n'.format(sync_reset))
        fh.write('    RSTC => {}, -- 1-bit input: Reset input for CREG\n'.format(sync_reset))
        fh.write('    RSTCTRL => {}, -- 1-bit input: Reset input for OPMODEREG and CARRYINSELREG\n'.format(sync_reset))
        fh.write('    RSTD => {}, -- 1-bit input: Reset input for DREG and ADREG\n'.format(sync_reset))
        fh.write('    RSTM => {}, -- 1-bit input: Reset input for MREG\n'.format(sync_reset))
        fh.write('    RSTP => {} -- 1-bit input: Reset input for PREG\n'.format(sync_reset))
        fh.write(');\n\n')
        fh.write('end rtl;\n')

    return file_name, module_name
