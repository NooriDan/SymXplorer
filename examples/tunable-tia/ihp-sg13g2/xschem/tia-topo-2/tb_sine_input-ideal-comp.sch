v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
L 4 40 -1160 350 -1160 {}
L 4 350 -1160 350 -940 {}
L 4 30 -940 350 -940 {}
L 4 30 -1160 30 -940 {}
L 4 30 -1160 40 -1160 {}
L 4 40 -780 620 -780 {}
L 4 620 -620 620 -380 {}
L 4 40 -380 620 -380 {}
L 4 40 -620 40 -380 {}
L 4 390 -1160 1130 -1160 {}
L 4 1130 -1160 1130 -860 {}
L 4 390 -860 1130 -860 {}
L 4 390 -1160 390 -860 {}
L 4 662.5 -620 960 -620 {}
L 4 960 -620 960 -386.25 {}
L 4 660 -620 660 -387.5 {}
L 4 660 -620 662.5 -620 {}
L 4 960 -386.25 960 -380 {}
L 4 660 -380 960 -380 {}
L 4 660 -387.5 660 -381.25 {}
L 4 660 -381.25 660 -380 {}
L 4 40 -770 40 -620 {}
L 4 40 -780 40 -770 {}
L 4 620 -780 620 -620 {}
B 2 1160 -790 1960 -390 {flags=graph
y1=-1.9305556
y2=-0.9305556
ypos1=0
ypos2=2
divy=5
subdivy=1
unity=1
x1=0
x2=1e-07
divx=5
subdivx=1
xlabmag=1.0
ylabmag=1.0
node=vop
color=4
dataset=-1
unitx=1
logx=0
logy=0
rawfile=$netlist_dir/rawspice.raw
sim_type=tran
autoload=1
rainbow=0
mode=Line}
T {Supply} 40 -1180 0 0 0.2 0.2 {}
T {Stimulus} 40 -800 0 0 0.2 0.2 {}
T {DUT} 400 -1180 0 0 0.2 0.2 {}
T {Load} 665 -635 0 0 0.2 0.2 {}
N 110 -1020 110 -1000 {
lab=GND}
N 110 -1120 110 -1080 {
lab=VSS}
N 190 -1020 190 -1000 {
lab=GND}
N 110 -1000 110 -980 {
lab=GND}
N 190 -1120 190 -1080 {
lab=VDD}
N 110 -1000 190 -1000 {
lab=GND}
N 340 -450 340 -430 {
lab=GND}
N 880 -1070 920 -1070 {lab=VDD}
N 880 -1050 920 -1050 {lab=VSS}
N 270 -1020 270 -1000 {lab=GND}
N 190 -1000 270 -1000 {lab=GND}
N 270 -1120 270 -1080 {lab=Vbias}
N 900 -1000 918.75 -1000 {lab=Von}
N 700 -476.25 700 -466.25 {lab=GND}
N 700 -476.25 780 -476.25 {lab=GND}
N 780 -486.25 780 -476.25 {lab=GND}
N 780 -566.25 780 -546.25 {lab=Vop}
N 700 -566.25 700 -546.25 {lab=Von}
N 700 -486.25 700 -476.25 {lab=GND}
N 510 -1081.25 510 -1060 {lab=Vbias}
N 490 -1081.25 510 -1081.25 {lab=Vbias}
N 510 -1020 510 -1000 {lab=Ip}
N 490 -1000 510 -1000 {lab=Ip}
N 472.5 -1040 582.5 -1040 {lab=In}
N 510 -1060 582.5 -1060 {lab=Vbias}
N 510 -1020 582.5 -1020 {lab=Ip}
N 150 -440 150 -420 {lab=GND}
N 340 -590 410 -590 {lab=#net1}
N 340 -450 410 -450 {lab=GND}
N 410 -590 410 -580 {lab=#net1}
N 410 -520 410 -510 {lab=#net2}
N 300 -590 340 -590 {lab=#net1}
N 300 -590 300 -580 {lab=#net1}
N 300 -520 300 -510 {lab=#net3}
N 300 -450 340 -450 {lab=GND}
N 150 -440 180 -440 {lab=GND}
N 80 -440 150 -440 {lab=GND}
N 180 -600 180 -580 {lab=#net4}
N 140 -600 180 -600 {lab=#net4}
N 80 -600 80 -580 {lab=#net4}
N 140 -640 140 -600 {lab=#net4}
N 80 -600 140 -600 {lab=#net4}
N 140 -750 140 -700 {lab=In}
N 340 -750 340 -700 {lab=Ip}
N 340 -640 340 -590 {lab=#net1}
N 180 -520 180 -500 {lab=#net5}
N 80 -520 80 -500 {lab=#net6}
N 900 -1000 900 -910 {lab=Von}
N 878.75 -1000 900 -1000 {lab=Von}
N 900 -910 980 -910 {lab=Von}
N 878.75 -1020 980 -1020 {lab=Vop}
N 980 -1020 980 -950 {lab=Vop}
C {devices/lab_wire.sym} 920 -1070 0 1 {name=p2 sig_type=std_logic lab=VDD}
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

    * print all

    write op.raw

    echo ----------------------------------
    echo END of OP
    echo ----------------------------------

    set appendwrite

    echo
    echo (B) start of TRAN Sim
    echo ----------------------------------

    echo starting the tranient sim...
    tran 0.01ns 100ns

    plot v(vop, von)
    plot Vmeas5#branch

    echo ----------------------------------
    echo END of TRAN sim
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
descr="Load waves TRAN" 
tclcommand="
xschem raw_read $netlist_dir/rawspice.raw tran

"
}
C {devices/vsource.sym} 110 -1050 0 0 {name=V0 value=0 savecurrent=false}
C {devices/gnd.sym} 110 -980 0 0 {name=l3 lab=GND}
C {devices/vsource.sym} 190 -1050 0 0 {name=V2 value=\{vdd\} savecurrent=false}
C {devices/lab_wire.sym} 110 -1120 0 0 {name=p1 sig_type=std_logic lab=VSS}
C {devices/gnd.sym} 340 -430 0 0 {name=l6 lab=GND}
C {devices/lab_wire.sym} 140 -750 0 0 {name=p12 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 190 -1120 0 0 {name=p5 sig_type=std_logic lab=VDD}
C {devices/lab_wire.sym} 920 -1050 0 1 {name=p9 sig_type=std_logic lab=VSS}
C {devices/code.sym} 30 -300 0 0 {name=MODELS only_toplevel=true
format="tcleval( @value )"
value="
********** Models ************
.lib cornerMOSlv.lib mos_tt
.lib cornerRES.lib res_typ 
.lib $::SG13G2_MODELS/cornerCAP.lib cap_typ
********** END of Models ************
"}
C {devices/vsource.sym} 270 -1050 0 0 {name=V1 value=\{vbias\} savecurrent=false}
C {devices/lab_wire.sym} 270 -1120 0 0 {name=p6 sig_type=std_logic lab=Vbias}
C {devices/lab_wire.sym} 490 -1081.25 0 0 {name=p7 sig_type=std_logic lab=Vbias}
C {devices/capa.sym} 700 -516.25 0 0 {name=C1
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_wire.sym} 918.75 -1020 0 1 {name=p3 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 918.75 -1000 0 1 {name=p4 sig_type=std_logic lab=Von}
C {devices/lab_wire.sym} 472.5 -1040 0 0 {name=p8 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 490 -1000 0 0 {name=p10 sig_type=std_logic lab=Ip}
C {devices/capa.sym} 780 -516.25 0 0 {name=C2
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/gnd.sym} 700 -466.25 0 0 {name=l1 lab=GND}
C {devices/lab_wire.sym} 780 -566.25 0 1 {name=p11 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 700 -566.25 0 1 {name=p14 sig_type=std_logic lab=Von}
C {devices/isource.sym} 410 -480 0 0 {name=Icm value=\{icm\}}
C {devices/gnd.sym} 150 -420 0 0 {name=l4 lab=GND}
C {devices/lab_wire.sym} 340 -750 0 0 {name=p13 sig_type=std_logic lab=Ip}
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
C {devices/code.sym} 270 -300 0 0 {name=tb_params only_toplevel=false value="
****** TB Parameters ****** 
.param icm=0
.param vdd=1.2
.param vbias=0.5
.param cload=20p
******  End of TB Params ****** 
"}
C {devices/code.sym} 150 -300 0 0 {name=dut_params only_toplevel=false value="
****** All DUT parameters ******

* MOSFET PARAMETERS
.param x_dut_nfet_w=5.0u
.param x_dut_nfet_l=0.5u
.param x_dut_nfet_m=10

* PASSIVE ELEMENT PARAMETERS

.param x_dut_cap_size=50p
.param x_dut_res_load_size=1k
.param x_dut_res_2_size=100.0k
.param x_dut_ind_size=1.0n

****** END of DUT Params ****** 
"}
C {tunable-tia/ihp-sg13g2/xschem/tia-topo-2/tia-bpf-2-ideal-comp.sym} 582.5 -970 0 0 {name=x_dut }
C {ammeter.sym} 410 -550 0 0 {name=Vmeas_icm savecurrent=true spice_ignore=0}
C {ammeter.sym} 140 -670 0 0 {name=Vmeas_in savecurrent=true spice_ignore=0}
C {cccs.sym} 180 -470 0 0 {name=F1 vnam=Vmeas_icm value=1}
C {devices/isource.sym} 300 -480 0 0 {name=Idm value="SIN(0, 10n, 10G)"}
C {ammeter.sym} 300 -550 0 0 {name=Vmeas_idm savecurrent=true spice_ignore=0}
C {cccs.sym} 80 -470 0 0 {name=F2 vnam=Vmeas_idm value=-1}
C {ammeter.sym} 80 -550 0 1 {name=Vmeas_idm_cccs savecurrent=true spice_ignore=0}
C {ammeter.sym} 180 -550 0 0 {name=Vmeas_icm_cccs savecurrent=true spice_ignore=0}
C {ammeter.sym} 340 -670 0 0 {name=Vmeas_ip savecurrent=true spice_ignore=0}
C {spice_probe_vdiff.sym} 980 -930 0 0 {name=p15}
C {ngspice_probe.sym} 980 -950 0 0 {name=r1}
C {ngspice_probe.sym} 980 -910 0 0 {name=r2}
C {spice_probe.sym} 980 -1020 0 0 {name=p16 attrs=""}
