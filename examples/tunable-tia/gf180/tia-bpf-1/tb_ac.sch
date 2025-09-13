v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
B 2 1100 20 1900 420 {flags=graph
y1=-17.690276
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
y2=43.309728
subdivy=1
mode=Line
x1=5.5993832
x2=10.399384}
B 2 1100 420 1900 820 {flags=graph
y1=-180.00001
y2=180.98548
ypos1=0
ypos2=2
divy=5
subdivy=1
unity=1
x1=5.4578915
x2=12.457891
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
autoload=1
rawfile=$netlist_dir/rawspice.raw}
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

} 1930 50 0 0 0.6 0.6 {floater=1}
N 50 170 50 190 {
lab=GND}
N 50 70 50 110 {
lab=VSS}
N 130 170 130 190 {
lab=GND}
N 50 190 50 210 {
lab=GND}
N 130 70 130 110 {
lab=VDD}
N 50 190 130 190 {
lab=GND}
N 220 540 220 560 {
lab=GND}
N 310 540 310 560 {
lab=GND}
N 690 80 730 80 {lab=VDD}
N 690 100 730 100 {lab=VSS}
N 210 170 210 190 {lab=GND}
N 130 190 210 190 {lab=GND}
N 210 70 210 110 {lab=Vbias}
N 360 140 390 140 {lab=Vbias}
N 690 120 730 120 {lab=Vop}
N 690 140 730 140 {lab=Von}
N 360 80 390 80 {lab=In}
N 360 100 390 100 {lab=Ip}
N 380 280 380 300 {lab=GND}
N 380 290 460 290 {lab=GND}
N 460 280 460 290 {lab=GND}
N 460 200 460 220 {lab=Vop}
N 380 200 380 220 {lab=Von}
N 220 460 220 480 {lab=In}
N 220 460 310 460 {lab=In}
N 310 460 310 480 {lab=In}
N 270 400 270 460 {lab=In}
N 40 540 40 560 {
lab=GND}
N 130 540 130 560 {
lab=GND}
N 40 460 40 480 {lab=Ip}
N 40 460 130 460 {lab=Ip}
N 130 460 130 480 {lab=Ip}
N 90 400 90 460 {lab=Ip}
C {devices/lab_wire.sym} 730 80 0 1 {name=p2 sig_type=std_logic lab=VDD}
C {devices/code.sym} 24.71697957735296 690 0 0 {name=controls
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
    let gm_id = @m.x_dut.xm1.m0[gm]/@m.x_dut.xm1.m0[id]
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
C {devices/launcher.sym} 900 720 0 0 {name=h26
descr="Annotate OP" 
tclcommand="
set show_hidden_texts 1; 
xschem annotate_op
"
}
C {devices/launcher.sym} 900 780 0 0 {name=h27
descr="Load waves AC" 
tclcommand="
xschem raw_read $netlist_dir/rawspice.raw ac

"
}
C {devices/vsource.sym} 50 140 0 0 {name=V0 value=0 savecurrent=false}
C {devices/gnd.sym} 50 210 0 0 {name=l3 lab=GND}
C {devices/vsource.sym} 130 140 0 0 {name=V2 value=\{vdd\} savecurrent=false}
C {devices/lab_wire.sym} 50 70 0 0 {name=p1 sig_type=std_logic lab=VSS}
C {devices/gnd.sym} 220 560 0 0 {name=l5 lab=GND}
C {devices/gnd.sym} 310 560 0 0 {name=l6 lab=GND}
C {devices/lab_wire.sym} 270 400 0 0 {name=p12 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 130 70 0 0 {name=p5 sig_type=std_logic lab=VDD}
C {devices/lab_wire.sym} 730 100 0 1 {name=p9 sig_type=std_logic lab=VSS}
C {devices/code.sym} 160 690 0 0 {name=MODELS only_toplevel=true
format="tcleval( @value )"
value="
.include $::180MCU_MODELS/design.ngspice
.lib $::180MCU_MODELS/sm141064.ngspice typical

.lib $::180MCU_MODELS/sm141064.ngspice res_typical
* .lib $::180MCU_MODELS/sm141064.ngspice res_statistical
"}
C {tia-bpf-1/bpf-tia-1-all-r-c.sym} 410 70 0 0 {name=x_dut}
C {devices/vsource.sym} 210 140 0 0 {name=V1 value=\{vbias\} savecurrent=false}
C {devices/lab_wire.sym} 210 70 0 0 {name=p6 sig_type=std_logic lab=Vbias}
C {devices/lab_wire.sym} 360 140 0 0 {name=p7 sig_type=std_logic lab=Vbias}
C {devices/capa.sym} 380 250 0 0 {name=C1
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_wire.sym} 730 120 0 1 {name=p3 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 730 140 0 1 {name=p4 sig_type=std_logic lab=Von}
C {devices/lab_wire.sym} 360 80 0 0 {name=p8 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 360 100 0 0 {name=p10 sig_type=std_logic lab=Ip}
C {devices/capa.sym} 460 250 0 0 {name=C2
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/gnd.sym} 380 300 0 0 {name=l1 lab=GND}
C {devices/lab_wire.sym} 460 200 0 1 {name=p11 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 380 200 0 1 {name=p14 sig_type=std_logic lab=Von}
C {devices/isource.sym} 310 510 0 0 {name=Icm value=\{icm\}}
C {devices/isource.sym} 220 510 0 0 {name=Idn value="ac -0.5"}
C {devices/gnd.sym} 40 560 0 0 {name=l2 lab=GND}
C {devices/gnd.sym} 130 560 0 0 {name=l4 lab=GND}
C {devices/lab_wire.sym} 90 400 0 0 {name=p13 sig_type=std_logic lab=Ip}
C {devices/isource.sym} 130 510 0 0 {name=Icm1 value=\{icm\}}
C {devices/isource.sym} 40 510 0 0 {name=Idp value="ac 0.5"}
C {devices/title.sym} 200 880 0 0 {name=l7 author="Danial NZ"}
C {devices/code.sym} 430 690 0 0 {name=save_ngspice only_toplevel=false value="

*.option savecurrents
* Saving gm values for transistors
.save @m.x_dut.xm1.m0[gm]
.save @m.x_dut.xm2.m0[gm]

* Saving id values for transistors
.save @m.x_dut.xm1.m0[id]
.save @m.x_dut.xm2.m0[id]

"}
C {devices/ngspice_get_expr.sym} 840 180 2 0 {name=r7 
node="[format %.2g [expr 1e6*[ngspice::get_node \{i(@m.x_dut.xm1.m0[id])\}]]]"
descr="id uA="}
C {devices/ngspice_get_expr.sym} 840 220 2 0 {name=r1 
node="[format %.2g [expr 1e3*[ngspice::get_node \{@m.x_dut.xm1.m0[gm]\}]]]"
descr="gm (mS) ="}
C {devices/ngspice_get_expr.sym} 930 180 2 0 {name=r2 
node="[format %.2g [expr [ngspice::get_node \{@m.x_dut.xm1.m0[gm]\}]/[ngspice::get_node \{i(@m.x_dut.xm1.m0[id])\}]]]"
descr="gm/id="
color='blue'}
C {devices/code.sym} 300 690 0 0 {name=tb_params only_toplevel=false value="
* TB Parameters
.param idd=1.8e-6
.param icm=0.5e-6
.param vdd=6.0
.param vbias=1.0
.param cload=1p
"}
C {devices/code.sym} 570 690 0 0 {name=dut_params only_toplevel=false value="

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
