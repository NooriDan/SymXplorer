v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
B 2 1160 -1560 1960 -1160 {flags=graph
y1=-0.41664606
y2=163.58335
ypos1=0
ypos2=2
divy=5
subdivy=4
unity=1
x2=21.07087
divx=5
xlabmag=1.0
ylabmag=1.0


dataset=-1
unitx=1
logx=0
logy=0
rainbow=0
digital=0
sim_type=tran
autoload=1
subdivx=4
x1=12.07087}
B 2 1160 -1160 1960 -760 {flags=graph
y1=89.324746
y2=450.31024
ypos1=0
ypos2=2
divy=5
subdivy=1
unity=1
x1=9.4472873
x2=16.447289
divx=5
subdivx=4
xlabmag=1.0
ylabmag=1.0
node="ph(Vout)"
color=5
dataset=-1
unitx=1
logx=0
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

} 1990 -1530 0 0 0.6 0.6 {floater=1}
N 30 -1510 30 -1490 {
lab=GND}
N 30 -1610 30 -1570 {
lab=VSS}
N 110 -1510 110 -1490 {
lab=GND}
N 30 -1490 30 -1470 {
lab=GND}
N 110 -1610 110 -1570 {
lab=VDD}
N 30 -1490 110 -1490 {
lab=GND}
N 200 -1140 200 -1120 {
lab=GND}
N 290 -1140 290 -1120 {
lab=GND}
N 770 -1550 810 -1550 {lab=VDD}
N 770 -1530 810 -1530 {lab=VSS}
N 190 -1510 190 -1490 {lab=GND}
N 110 -1490 190 -1490 {lab=GND}
N 190 -1610 190 -1570 {lab=Vbias}
N 440 -1490 470 -1490 {lab=Vbias}
N 770 -1510 810 -1510 {lab=Vop}
N 770 -1490 810 -1490 {lab=Von}
N 440 -1550 470 -1550 {lab=In}
N 440 -1530 470 -1530 {lab=Ip}
N 560 -1340 560 -1320 {lab=GND}
N 560 -1330 640 -1330 {lab=GND}
N 640 -1340 640 -1330 {lab=GND}
N 640 -1420 640 -1400 {lab=Vop}
N 560 -1420 560 -1400 {lab=Von}
N 200 -1220 200 -1200 {lab=In}
N 200 -1220 290 -1220 {lab=In}
N 290 -1220 290 -1200 {lab=In}
N 250 -1280 250 -1220 {lab=In}
N 20 -1140 20 -1120 {
lab=GND}
N 110 -1140 110 -1120 {
lab=GND}
N 20 -1220 20 -1200 {lab=Ip}
N 20 -1220 110 -1220 {lab=Ip}
N 110 -1220 110 -1200 {lab=Ip}
N 70 -1280 70 -1220 {lab=Ip}
C {devices/lab_wire.sym} 810 -1550 0 1 {name=p2 sig_type=std_logic lab=VDD}
C {devices/code.sym} 14.71697957735296 -850 0 0 {name=controls
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

    print all
    write op.raw

    echo ----------------------------------
    echo END of OP
    echo ----------------------------------

    * set appendwrite
echo
    echo (B) start of DC Sweep
    echo ----------------------------------

    * example: sweep VIN from 0V to 1.8V in 0.01V steps
    *dc VIN 0 1.8 0.01

    * we sweep the differential current
    dc ICM 0 1.8u 0.01


    * post-processing
    let Vout_dc = v(Vop) - v(Von)

    * The following works if the DUT is voltage follower and configured in unity-gain
    *meas dc Vout_mid find Vout_dc when VIN=0.9
    *meas dc Vin_th when Vout_dc=0.9 cross=1
    plot Vout_dc vs ip

    echo ----------------------------------
    echo END of DC Sweep
    echo ----------------------------------

    echo
    echo (C) start of Transient Sim (step input)
    echo ----------------------------------

    * step input should be defined in netlist, e.g.:
    * VIN in 0 pulse(0 1.8 0 1n 1n 1u 2u)
    tran 1n 10u

    * post-processing
    let Vout_tran = v(Vop) - v(Von)
    meas tran t_delay trig v(Vin) val=0.9 rise=1 targ Vout_tran val=0.9 rise=1
    plot Vout_tran

    echo ----------------------------------
    echo END of Transient Step Sim
    echo ----------------------------------

    echo
    echo (D) start of Transient Sim (sinusoidal input)
    echo ----------------------------------

    * sinusoidal input should be defined in netlist, e.g.:
    * VIN in 0 sin(0 0.5 1e6 0 0)
    tran 1n 10u

    * post-processing
    let Vout_sin = v(Vop) - v(Von)
    fft Vout_sin
    plot Vout_sin

    echo ----------------------------------
    echo END of Transient Sin Sim
    echo ----------------------------------

    echo SAVING SIMULATION RESULTS IN THE RAW FILE
    save all
    write

    echo ==================================
    echo END of Control
    echo ==================================
.endc
"}
C {devices/launcher.sym} 960 -840 0 0 {name=h26
descr="Annotate OP" 
tclcommand="
set show_hidden_texts 1; 
xschem annotate_op
"
}
C {devices/launcher.sym} 960 -780 0 0 {name=h27
descr="Load waves AC" 
tclcommand="
xschem raw_read $netlist_dir/rawspice.raw ac

"
}
C {devices/vsource.sym} 30 -1540 0 0 {name=V0 value=0 savecurrent=false}
C {devices/gnd.sym} 30 -1470 0 0 {name=l3 lab=GND}
C {devices/vsource.sym} 110 -1540 0 0 {name=V2 value=\{vdd\} savecurrent=false}
C {devices/lab_wire.sym} 30 -1610 0 0 {name=p1 sig_type=std_logic lab=VSS}
C {devices/gnd.sym} 200 -1120 0 0 {name=l5 lab=GND}
C {devices/gnd.sym} 290 -1120 0 0 {name=l6 lab=GND}
C {devices/lab_wire.sym} 250 -1280 0 0 {name=p12 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 110 -1610 0 0 {name=p5 sig_type=std_logic lab=VDD}
C {devices/lab_wire.sym} 810 -1530 0 1 {name=p9 sig_type=std_logic lab=VSS}
C {devices/code.sym} 450 -850 0 0 {name=MODELS only_toplevel=true
format="tcleval( @value )"
value="
.include $::180MCU_MODELS/design.ngspice
.lib $::180MCU_MODELS/sm141064.ngspice typical

.lib $::180MCU_MODELS/sm141064.ngspice res_typical
* .lib $::180MCU_MODELS/sm141064.ngspice res_statistical
"}
C {tia-bpf-1/bpf-tia-1-all-r-c.sym} 490 -1560 0 0 {name=x_dut}
C {devices/vsource.sym} 190 -1540 0 0 {name=V1 value=\{vbias\} savecurrent=false}
C {devices/lab_wire.sym} 190 -1610 0 0 {name=p6 sig_type=std_logic lab=Vbias}
C {devices/lab_wire.sym} 440 -1490 0 0 {name=p7 sig_type=std_logic lab=Vbias}
C {devices/capa.sym} 560 -1370 0 0 {name=C1
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_wire.sym} 810 -1510 0 1 {name=p3 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 810 -1490 0 1 {name=p4 sig_type=std_logic lab=Von}
C {devices/lab_wire.sym} 440 -1550 0 0 {name=p8 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 440 -1530 0 0 {name=p10 sig_type=std_logic lab=Ip}
C {devices/capa.sym} 640 -1370 0 0 {name=C2
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/gnd.sym} 560 -1320 0 0 {name=l1 lab=GND}
C {devices/lab_wire.sym} 640 -1420 0 1 {name=p11 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 560 -1420 0 1 {name=p14 sig_type=std_logic lab=Von}
C {devices/isource.sym} 290 -1170 0 0 {name=Icm value=\{icm\}}
C {devices/isource.sym} 200 -1170 0 0 {name=Idn value="dc 0 ac 0"}
C {devices/gnd.sym} 20 -1120 0 0 {name=l2 lab=GND}
C {devices/gnd.sym} 110 -1120 0 0 {name=l4 lab=GND}
C {devices/lab_wire.sym} 70 -1280 0 0 {name=p13 sig_type=std_logic lab=Ip}
C {devices/isource.sym} 110 -1170 0 0 {name=Icm1 value=\{icm\}}
C {devices/isource.sym} 20 -1170 0 0 {name=Idp value="PULSE(0  \{i_step\} 100n 1n 1n 1u 2u)"}
C {devices/title.sym} 190 -690 0 0 {name=l7 author="Danial NZ"}
C {devices/code.sym} 590 -850 0 0 {name=save_ngspice only_toplevel=false value="

*.option savecurrents
* Saving gm values for transistors
.save @m.x_dut.xm1.m0[gm]
.save @m.x_dut.xm2.m0[gm]

* Saving id values for transistors
.save @m.x_dut.xm1.m0[id]
.save @m.x_dut.xm2.m0[id]

"}
C {devices/ngspice_get_expr.sym} 950 -1570 2 0 {name=r7 
node="[format %.2g [expr 1e6*[ngspice::get_node \{i(@m.x_dut.xm1.m0[id])\}]]]"
descr="id uA="}
C {devices/ngspice_get_expr.sym} 950 -1530 2 0 {name=r1 
node="[format %.2g [expr 1e3*[ngspice::get_node \{@m.x_dut.xm1.m0[gm]\}]]]"
descr="gm (mS) ="}
C {devices/ngspice_get_expr.sym} 1040 -1570 2 0 {name=r2 
node="[format %.2g [expr [ngspice::get_node \{@m.x_dut.xm1.m0[gm]\}]/[ngspice::get_node \{i(@m.x_dut.xm1.m0[id])\}]]]"
descr="gm/id="
color='blue'}
C {devices/code.sym} 310 -850 0 0 {name=tb_params only_toplevel=false value="
.param icm=0.5e-6
.param i_step=0.5e-6

.param vdd=6.0
.param vbias=1.0

.param cload=1p
"}
C {devices/code.sym} 150 -850 0 0 {name=dut_params only_toplevel=false value="

* All DUT parameters

* MOSFET PARAMETERS
.param x_dut_m1_2_w=0.22u
.param x_dut_m1_2_l=0.40u
.param x_dut_m1_2_nf=2

* PASSIVE ELEMENT PARAMETERS
.param x_dut_ldd=2n
.param x_dut_cdd=1n

.param x_dut_rs=1k

.param x_dut_r4=1k
"}
