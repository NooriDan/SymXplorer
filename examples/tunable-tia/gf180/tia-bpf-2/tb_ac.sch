v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
L 4 940 -1540 1130 -1540 {}
L 4 1130 -1540 1130 -1460 {}
L 4 940 -1460 1130 -1460 {}
L 4 940 -1540 940 -1460 {}
L 4 940 -1420 1130 -1420 {}
L 4 1130 -1420 1130 -1340 {}
L 4 940 -1340 1130 -1340 {}
L 4 940 -1420 940 -1340 {}
B 2 1240 -1590 2040 -1190 {flags=graph
y1=58.027054
y2=189.22709
ypos1=0
ypos2=2
divy=5
subdivy=4
unity=1
x2=12.929322
divx=5
xlabmag=1.0
ylabmag=1.0


dataset=-1
unitx=1
logx=1
logy=0
color=4
node=vout_mag_db
rainbow=0
rawfile=$netlist_dir/rawspice.raw
digital=0
autoload=1
subdivx=10
x1=5.7293217}
B 2 1240 -1190 2040 -790 {flags=graph
y1=-180.00001
y2=180.98548
ypos1=0
ypos2=2
divy=5
subdivy=1
unity=1
x1=5.7293217
x2=12.929322
divx=5
subdivx=10
xlabmag=1.0
ylabmag=1.0
node="ph(Vout)"
color=5
dataset=-1
unitx=1
logx=1
logy=0
autoload=1}
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

} 2070 -1560 0 0 0.6 0.6 {floater=1}
T {PMOS} 1070 -1460 0 0 0.4 0.4 {}
T {NMOS} 1070 -1340 0 0 0.4 0.4 {}
T {GM/ID Values} 960 -1580 0 0 0.4 0.4 {}
N 40 -1490 40 -1470 {
lab=GND}
N 40 -1590 40 -1550 {
lab=VSS}
N 120 -1490 120 -1470 {
lab=GND}
N 40 -1470 40 -1450 {
lab=GND}
N 120 -1590 120 -1550 {
lab=VDD}
N 40 -1470 120 -1470 {
lab=GND}
N 210 -1120 210 -1100 {
lab=GND}
N 300 -1120 300 -1100 {
lab=GND}
N 780 -1530 820 -1530 {lab=VDD}
N 780 -1510 820 -1510 {lab=VSS}
N 200 -1490 200 -1470 {lab=GND}
N 120 -1470 200 -1470 {lab=GND}
N 200 -1590 200 -1550 {lab=Vbias}
N 450 -1470 480 -1470 {lab=Ip}
N 780 -1490 820 -1490 {lab=Vop}
N 780 -1470 820 -1470 {lab=Von}
N 450 -1530 480 -1530 {lab=Vr}
N 450 -1510 480 -1510 {lab=Vbias}
N 570 -1320 570 -1300 {lab=GND}
N 570 -1310 650 -1310 {lab=GND}
N 650 -1320 650 -1310 {lab=GND}
N 650 -1400 650 -1380 {lab=Vop}
N 570 -1400 570 -1380 {lab=Von}
N 210 -1200 210 -1180 {lab=In}
N 210 -1200 300 -1200 {lab=In}
N 300 -1200 300 -1180 {lab=In}
N 260 -1260 260 -1200 {lab=In}
N 30 -1120 30 -1100 {
lab=GND}
N 120 -1120 120 -1100 {
lab=GND}
N 30 -1200 30 -1180 {lab=Ip}
N 30 -1200 120 -1200 {lab=Ip}
N 120 -1200 120 -1180 {lab=Ip}
N 80 -1260 80 -1200 {lab=Ip}
N 450 -1490 480 -1490 {lab=In}
N 300 -1490 300 -1470 {lab=GND}
N 200 -1470 300 -1470 {lab=GND}
N 300 -1590 300 -1550 {lab=Vr}
C {devices/lab_wire.sym} 820 -1530 0 1 {name=p2 sig_type=std_logic lab=VDD}
C {devices/code.sym} 24.71697957735296 -930 0 0 {name=controls
simulator=ngspice
only_toplevel=false
value="
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
    let gm_id_1_2 = @m.x_dut.xm1.m0[gm]/@m.x_dut.xm1.m0[id]
    let gm_id_3_4 = @m.x_dut.xm3.m0[gm]/@m.x_dut.xm3.m0[id]

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
    ac dec 200 1e1 1e12

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
    meas ac FC when vout_mag=GMAX

    let gmax_3db = GMAX/sqrt(2)

    meas ac FLO when vout_mag=gmax_3db cross=1
    meas ac FHI when vout_mag=gmax_3db cross=2

    let BW = FHI-FLO
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
"}
C {devices/launcher.sym} 1020 -900 0 0 {name=h26
descr="Annotate OP" 
tclcommand="
set show_hidden_texts 1; 
xschem annotate_op
"
}
C {devices/launcher.sym} 1020 -840 0 0 {name=h27
descr="Load waves AC" 
tclcommand="
xschem raw_read $netlist_dir/rawspice.raw ac

"
}
C {devices/vsource.sym} 40 -1520 0 0 {name=V0 value=0 savecurrent=false}
C {devices/gnd.sym} 40 -1450 0 0 {name=l3 lab=GND}
C {devices/vsource.sym} 120 -1520 0 0 {name=V2 value=\{vdd\} savecurrent=false}
C {devices/lab_wire.sym} 40 -1590 0 0 {name=p1 sig_type=std_logic lab=VSS}
C {devices/gnd.sym} 210 -1100 0 0 {name=l5 lab=GND}
C {devices/gnd.sym} 300 -1100 0 0 {name=l6 lab=GND}
C {devices/lab_wire.sym} 260 -1260 0 0 {name=p12 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 120 -1590 0 0 {name=p5 sig_type=std_logic lab=VDD}
C {devices/lab_wire.sym} 820 -1510 0 1 {name=p9 sig_type=std_logic lab=VSS}
C {devices/code.sym} 150 -930 0 0 {name=MODELS only_toplevel=true
format="tcleval( @value )"
value="
.include $::180MCU_MODELS/design.ngspice
.lib $::180MCU_MODELS/sm141064.ngspice typical

.lib $::180MCU_MODELS/sm141064.ngspice res_typical
* .lib $::180MCU_MODELS/sm141064.ngspice res_statistical
"}
C {tia-bpf-2/bpf-tia-2.sym} 500 -1540 0 0 {name=x_dut}
C {devices/vsource.sym} 200 -1520 0 0 {name=V1 value=\{vbias\} savecurrent=false}
C {devices/lab_wire.sym} 200 -1590 0 0 {name=p6 sig_type=std_logic lab=Vbias}
C {devices/lab_wire.sym} 450 -1510 0 0 {name=p7 sig_type=std_logic lab=Vbias}
C {devices/capa.sym} 570 -1350 0 0 {name=C1
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_wire.sym} 820 -1490 0 1 {name=p3 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 820 -1470 0 1 {name=p4 sig_type=std_logic lab=Von}
C {devices/lab_wire.sym} 450 -1490 0 0 {name=p8 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 450 -1470 0 0 {name=p10 sig_type=std_logic lab=Ip}
C {devices/capa.sym} 650 -1350 0 0 {name=C2
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/gnd.sym} 570 -1300 0 0 {name=l1 lab=GND}
C {devices/lab_wire.sym} 650 -1400 0 1 {name=p11 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 570 -1400 0 1 {name=p14 sig_type=std_logic lab=Von}
C {devices/isource.sym} 300 -1150 0 0 {name=Icm value=\{icm\}}
C {devices/isource.sym} 210 -1150 0 0 {name=Idn value="ac -0.5"}
C {devices/gnd.sym} 30 -1100 0 0 {name=l2 lab=GND}
C {devices/gnd.sym} 120 -1100 0 0 {name=l4 lab=GND}
C {devices/lab_wire.sym} 80 -1260 0 0 {name=p13 sig_type=std_logic lab=Ip}
C {devices/isource.sym} 120 -1150 0 0 {name=Icm1 value=\{icm\}}
C {devices/isource.sym} 30 -1150 0 0 {name=Idp value="ac 0.5"}
C {devices/title.sym} 200 -770 0 0 {name=l7 author="Danial NZ"}
C {devices/code.sym} 510 -930 0 0 {name=save_ngspice only_toplevel=false value="

*.option savecurrents
* Saving gm values for transistors
.save @m.x_dut.xm1.m0[gm]
.save @m.x_dut.xm2.m0[gm]
.save @m.x_dut.xm3.m0[gm]
.save @m.x_dut.xm4.m0[gm]

* Saving id values for transistors
.save @m.x_dut.xm1.m0[id]
.save @m.x_dut.xm2.m0[id]
.save @m.x_dut.xm3.m0[id]
.save @m.x_dut.xm4.m0[id]

"}
C {devices/ngspice_get_expr.sym} 1010 -1410 2 0 {name=r7 
node="[format %.2g [expr 1e6*[ngspice::get_node \{i(@m.x_dut.xm1.m0[id])\}]]]"
descr="id uA="}
C {devices/ngspice_get_expr.sym} 1010 -1370 2 0 {name=r1 
node="[format %.2g [expr 1e3*[ngspice::get_node \{@m.x_dut.xm1.m0[gm]\}]]]"
descr="gm (mS) ="}
C {devices/ngspice_get_expr.sym} 1100 -1410 2 0 {name=r2 
node="[format %.2g [expr [ngspice::get_node \{@m.x_dut.xm1.m0[gm]\}]/[ngspice::get_node \{i(@m.x_dut.xm1.m0[id])\}]]]"
descr="gm/id="
color='blue'}
C {devices/code.sym} 390 -930 0 0 {name=tb_params only_toplevel=false value="
.param idd=1.8e-6
.param icm=0.5e-6
.param vdd=6.0

.param vbias=1.0
.param vr=1.5


.param cload=1p
"}
C {devices/code.sym} 270 -930 0 0 {name=dut_params only_toplevel=false value="

* All DUT parameters

* MOSFET PARAMETERS
.param x_dut_m1_2_w=0.22u
.param x_dut_m1_2_l=0.40u
.param x_dut_m1_2_nf=2

.param x_dut_m3_4_w=0.22u
.param x_dut_m3_4_l=0.40u
.param x_dut_m3_4_nf=2

* PASSIVE ELEMENT PARAMETERS
.param x_dut_cdd=1n

.param x_dut_lss=1k
"}
C {devices/vsource.sym} 300 -1520 0 0 {name=V3 value=\{vr\} savecurrent=false}
C {devices/lab_wire.sym} 300 -1590 0 0 {name=p15 sig_type=std_logic lab=Vr}
C {devices/lab_wire.sym} 450 -1530 0 0 {name=p16 sig_type=std_logic lab=Vr}
C {devices/ngspice_get_expr.sym} 1010 -1530 2 0 {name=r3 
node="[format %.2g [expr 1e6*[ngspice::get_node \{i(@m.x_dut.xm3.m0[id])\}]]]"
descr="id uA="}
C {devices/ngspice_get_expr.sym} 1010 -1490 2 0 {name=r4 
node="[format %.2g [expr 1e3*[ngspice::get_node \{@m.x_dut.xm3.m0[gm]\}]]]"
descr="gm (mS) ="}
C {devices/ngspice_get_expr.sym} 1100 -1530 2 0 {name=r5 
node="[format %.2g [expr [ngspice::get_node \{@m.x_dut.xm3.m0[gm]\}]/[ngspice::get_node \{i(@m.x_dut.xm3.m0[id])\}]]]"
descr="gm/id="
color='blue'}
