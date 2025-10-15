v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
L 4 50 -960 360 -960 {}
L 4 360 -960 360 -740 {}
L 4 40 -740 360 -740 {}
L 4 40 -960 40 -740 {}
L 4 40 -960 50 -960 {}
L 4 40 -620 620 -620 {}
L 4 620 -620 620 -380 {}
L 4 40 -380 620 -380 {}
L 4 40 -620 40 -380 {}
L 4 400 -960 1140 -960 {}
L 4 1140 -960 1140 -660 {}
L 4 400 -660 1140 -660 {}
L 4 400 -960 400 -660 {}
L 4 662.5 -620 960 -620 {}
L 4 960 -620 960 -386.25 {}
L 4 660 -620 660 -387.5 {}
L 4 660 -620 662.5 -620 {}
L 4 960 -386.25 960 -380 {}
L 4 660 -380 960 -380 {}
L 4 660 -387.5 660 -381.25 {}
L 4 660 -381.25 660 -380 {}
B 2 1170 -970 1970 -570 {flags=graph
y1=-26.331949
ypos1=0
ypos2=2
unity=1
unitx=1
logx=1
logy=0
color=4
node=vout_mag
rainbow=0
rawfile=$netlist_dir/rawspice.raw
digital=0
sim_type=ac
autoload=1
subdivx=10
divy=10
y2=34.668051
subdivy=1
mode=Line
x1=5
x2=12}
B 2 1170 -570 1970 -170 {flags=graph
y1=-161.19866
y2=199.78683
ypos1=0
ypos2=2
divy=5
subdivy=1
unity=1
x1=5
x2=12
divx=5
subdivx=10
xlabmag=1.0
ylabmag=1.0
dataset=-1
unitx=1
logx=1
logy=0
autoload=1
rawfile=$netlist_dir/rawspice.raw
sim_type=ac
color=4
node=phase(vout)}
T {tcleval(A0: [to_eng [xschem raw value A0 0]]
UGF: [to_eng [xschem raw value ugf 0]]
PM: [to_eng [xschem raw value pm 0]]
=======
max gain (dB): 
[to_eng [xschem raw value gmax_3db 0]]
--------------
f_center (Hz): 
[to_eng [xschem raw value FC 0]]
flo [to_eng [xschem raw value FLO 0]]
fhi [to_eng [xschem raw value FHI 0]]
--------------
Bandwidth (dB): 
[to_eng [xschem raw value BW 0]]
--------------
Q-factor: 
[to_eng [xschem raw value Q 0]]
========
)

} 2000 -940 0 0 0.6 0.6 {floater=1}
T {Supply} 50 -980 0 0 0.2 0.2 {}
T {Stimulus} 50 -640 0 0 0.2 0.2 {}
T {DUT} 410 -980 0 0 0.2 0.2 {}
T {Load} 665 -635 0 0 0.2 0.2 {}
N 120 -820 120 -800 {
lab=GND}
N 120 -920 120 -880 {
lab=VSS}
N 200 -820 200 -800 {
lab=GND}
N 120 -800 120 -780 {
lab=GND}
N 200 -920 200 -880 {
lab=VDD}
N 120 -800 200 -800 {
lab=GND}
N 290 -450 290 -430 {
lab=GND}
N 380 -450 380 -430 {
lab=GND}
N 890 -870 930 -870 {lab=VDD}
N 890 -850 930 -850 {lab=VSS}
N 280 -820 280 -800 {lab=GND}
N 200 -800 280 -800 {lab=GND}
N 280 -920 280 -880 {lab=Vbias}
N 888.75 -820 928.75 -820 {lab=Vop}
N 888.75 -800 928.75 -800 {lab=Von}
N 700 -476.25 700 -466.25 {lab=GND}
N 700 -476.25 780 -476.25 {lab=GND}
N 780 -486.25 780 -476.25 {lab=GND}
N 780 -566.25 780 -546.25 {lab=Vop}
N 700 -566.25 700 -546.25 {lab=Von}
N 290 -530 290 -510 {lab=In}
N 340 -530 380 -530 {lab=In}
N 380 -530 380 -510 {lab=In}
N 340 -590 340 -530 {lab=In}
N 110 -450 110 -430 {
lab=GND}
N 200 -450 200 -430 {
lab=GND}
N 110 -530 110 -510 {lab=Ip}
N 160 -530 200 -530 {lab=Ip}
N 200 -530 200 -510 {lab=Ip}
N 160 -590 160 -530 {lab=Ip}
N 700 -486.25 700 -476.25 {lab=GND}
N 290 -530 340 -530 {lab=In}
N 110 -530 160 -530 {lab=Ip}
N 520 -881.25 520 -860 {lab=Vbias}
N 500 -881.25 520 -881.25 {lab=Vbias}
N 520 -820 520 -800 {lab=Ip}
N 500 -800 520 -800 {lab=Ip}
N 482.5 -840 592.5 -840 {lab=In}
N 520 -860 592.5 -860 {lab=Vbias}
N 520 -820 592.5 -820 {lab=Ip}
C {devices/lab_wire.sym} 930 -870 0 1 {name=p2 sig_type=std_logic lab=VDD}
C {devices/code.sym} 514.716979577353 -300 0 0 {name=controls
simulator=ngspice
only_toplevel=false
value="
****** START OF CONTROL SECTION ****** 


.control

    echo ==================================
    echo start of Control
    echo ==================================

    save all

    echo
    echo (A) start of OP
    echo ----------------------------------

    * operating point
    op

    *let gm = n.x_dut.xm1.nsg13_lv_nmos[gm]
    *let id = n.x_dut.xm1.nsg13_lv_nmos[id]
    *let gmid = gm/id

    print all

    write op.raw

    echo ----------------------------------
    echo END of OP
    echo ----------------------------------

    set appendwrite

    echo
    echo (B) start of AC Sim
    echo ----------------------------------

    * run ac simulation
    ac dec 200 1e5 1e12

    echo (1) post processing the AC sim
    * magnitude
    let Vout = v(Vop) - v(Von)
    let vout_mag = abs(v(Vout))
    let vout_mag_db = 20*log(vout_mag)/log(10)

    * raw phase (between -180 and 180)
    let vout_phase = phase(v(Vout)) * 180/pi

    echo (2) measuring A0, PM, UGF...
    
    * Bandpass response parameters
    meas ac GMAX max vout_mag

    let gmax_3db = GMAX/sqrt(2)

    meas ac FLO when vout_mag=gmax_3db cross=1
    meas ac FHI when vout_mag=gmax_3db cross=2

    let BW = FHI-FLO
    let FC = (FHI+FLO)/2.0
    let Q  = FC/BW
    let gain_db = 20 * log(GMAX) / log(10)
    print BW Q gain_db
    *meas ac A0 find vout_mag at=1k
    meas ac PM find vout_phase when vout_mag=1
    meas ac UGF when vout_mag=1 fall=1

    echo (3) plotting...
    plot vout_mag

    echo ----------------------------------
    echo END of AC sim
    echo  ----------------------------------

    echo SAVING SIMULATION RESULTS IN THE RAW FILE
    save all
    write

    echo ==================================
    echo END of Control
    echo ==================================
.endc
******  END OF CONTROL SECTION ****** 
"}
C {devices/launcher.sym} 968.75 -271.25 0 0 {name=h26
descr="Annotate OP" 
tclcommand="
set show_hidden_texts 1; 
xschem annotate_op
"
}
C {devices/launcher.sym} 970 -210 0 0 {name=h27
descr="Load waves AC" 
tclcommand="
xschem raw_read $netlist_dir/rawspice.raw ac

"
}
C {devices/vsource.sym} 120 -850 0 0 {name=V0 value=0 savecurrent=false}
C {devices/gnd.sym} 120 -780 0 0 {name=l3 lab=GND}
C {devices/vsource.sym} 200 -850 0 0 {name=V2 value=\{vdd\} savecurrent=false}
C {devices/lab_wire.sym} 120 -920 0 0 {name=p1 sig_type=std_logic lab=VSS}
C {devices/gnd.sym} 290 -430 0 0 {name=l5 lab=GND}
C {devices/gnd.sym} 380 -430 0 0 {name=l6 lab=GND}
C {devices/lab_wire.sym} 340 -590 0 0 {name=p12 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 200 -920 0 0 {name=p5 sig_type=std_logic lab=VDD}
C {devices/lab_wire.sym} 930 -850 0 1 {name=p9 sig_type=std_logic lab=VSS}
C {devices/code.sym} 30 -300 0 0 {name=MODELS only_toplevel=true
format="tcleval( @value )"
value="
********** Models ************
.lib cornerMOSlv.lib mos_tt
.lib cornerRES.lib res_typ 
.lib $::SG13G2_MODELS/cornerCAP.lib cap_typ
********** END of Models ************
"}
C {devices/vsource.sym} 280 -850 0 0 {name=V1 value=\{vbias\} savecurrent=false}
C {devices/lab_wire.sym} 280 -920 0 0 {name=p6 sig_type=std_logic lab=Vbias}
C {devices/lab_wire.sym} 500 -881.25 0 0 {name=p7 sig_type=std_logic lab=Vbias}
C {devices/capa.sym} 700 -516.25 0 0 {name=C1
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_wire.sym} 928.75 -820 0 1 {name=p3 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 928.75 -800 0 1 {name=p4 sig_type=std_logic lab=Von}
C {devices/lab_wire.sym} 482.5 -840 0 0 {name=p8 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 500 -800 0 0 {name=p10 sig_type=std_logic lab=Ip}
C {devices/capa.sym} 780 -516.25 0 0 {name=C2
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/gnd.sym} 700 -466.25 0 0 {name=l1 lab=GND}
C {devices/lab_wire.sym} 780 -566.25 0 1 {name=p11 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 700 -566.25 0 1 {name=p14 sig_type=std_logic lab=Von}
C {devices/isource.sym} 380 -480 0 0 {name=Icm value=\{icm\}}
C {devices/isource.sym} 290 -480 0 0 {name=Idn value="dc 0 ac -0.5"}
C {devices/gnd.sym} 110 -430 0 0 {name=l2 lab=GND}
C {devices/gnd.sym} 200 -430 0 0 {name=l4 lab=GND}
C {devices/lab_wire.sym} 160 -590 0 0 {name=p13 sig_type=std_logic lab=Ip}
C {devices/isource.sym} 200 -480 0 0 {name=Icm1 value=\{icm\}}
C {devices/isource.sym} 110 -480 0 0 {name=Idp value="dc 0 ac 0.5"}
C {devices/title.sym} 270 -110 0 0 {name=l7 author="Danial NZ"}
C {devices/code.sym} 387.5 -300 0 0 {name=save_ngspice only_toplevel=false value="

****** Save Statements ****** 
*.option savecurrents

** nfet small signal parameters **
.save @n.x_dut.xm1.nsg13_lv_nmos[cdd]
.save @n.x_dut.xm1.nsg13_lv_nmos[cgb]
.save @n.x_dut.xm1.nsg13_lv_nmos[cgd]
.save @n.x_dut.xm1.nsg13_lv_nmos[cgdol]
.save @n.x_dut.xm1.nsg13_lv_nmos[cgg]
.save @n.x_dut.xm1.nsg13_lv_nmos[cgs]
.save @n.x_dut.xm1.nsg13_lv_nmos[cgsol]
.save @n.x_dut.xm1.nsg13_lv_nmos[cjd]
.save @n.x_dut.xm1.nsg13_lv_nmos[cjs]
.save @n.x_dut.xm1.nsg13_lv_nmos[css]
.save @n.x_dut.xm1.nsg13_lv_nmos[gds]
.save @n.x_dut.xm1.nsg13_lv_nmos[gm]
.save @n.x_dut.xm1.nsg13_lv_nmos[gmb]
.save @n.x_dut.xm1.nsg13_lv_nmos[ids]
.save @n.x_dut.xm1.nsg13_lv_nmos[l]
.save @n.x_dut.xm1.nsg13_lv_nmos[sfl]
.save @n.x_dut.xm1.nsg13_lv_nmos[sid]
.save @n.x_dut.xm1.nsg13_lv_nmos[vth]

.save @n.x_dut.xm1.nsg13_lv_nmos[vgs]
.save @n.x_dut.xm1.nsg13_lv_nmos[vds]
.save @n.x_dut.xm1.nsg13_lv_nmos[vdss]

******  End of Save Statements ****** 
"}
C {devices/code.sym} 260 -300 0 0 {name=tb_params only_toplevel=false value="
****** TB Parameters ****** 
.param icm=0.5e-6
.param vdd=1.2
.param vbias=0.5
.param cload=100p
******  End of TB Params ****** 
"}
C {devices/code.sym} 150 -300 0 0 {name=dut_params only_toplevel=false value="
****** All DUT parameters ******

* MOSFET PARAMETERS
.param x_dut_nfet_w=0.40u
.param x_dut_nfet_l=0.26u
.param x_dut_nfet_m=10

* PASSIVE ELEMENT PARAMETERS

.param x_dut_cap_size=1.0n
.param x_dut_res_size=10.0k
.param x_dut_ind_size=1.0n

****** END of DUT Params ****** 
"}
C {tunable-tia/ihp-sg13g2/xschem/tia-topo-3/tia-bpf-3-ideal-component.sym} 592.5 -770 0 0 {name=x_dut }
