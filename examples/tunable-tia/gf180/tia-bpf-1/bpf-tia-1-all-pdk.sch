v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
N 750 -420 780 -420 {lab=vop}
N 390 -420 420 -420 {lab=von}
N 360 -290 420 -290 {lab=inp}
N 750 -290 810 -290 {lab=inn}
N 90 -600 130 -600 {lab=vss}
N 90 -640 130 -640 {lab=vdd}
N 460 -340 710 -340 {lab=vbias}
N 340 -340 420 -340 {lab=vss}
N 750 -340 840 -340 {lab=vss}
N 420 -310 420 -260 {lab=inp}
N 750 -310 750 -270 {lab=inn}
N 420 -200 420 -180 {lab=vss}
N 750 -210 750 -190 {lab=vss}
N 440 -480 440 -460 {lab=von}
N 380 -460 440 -460 {lab=von}
N 380 -480 380 -460 {lab=von}
N 380 -550 380 -540 {lab=vdd}
N 380 -550 440 -550 {lab=vdd}
N 440 -550 440 -540 {lab=vdd}
N 400 -580 400 -550 {lab=vdd}
N 420 -460 420 -370 {lab=von}
N 700 -550 700 -540 {lab=vdd}
N 700 -550 770 -550 {lab=vdd}
N 770 -550 770 -540 {lab=vdd}
N 720 -580 720 -550 {lab=vdd}
N 700 -480 700 -460 {lab=vop}
N 700 -460 770 -460 {lab=vop}
N 770 -480 770 -460 {lab=vop}
N 750 -460 750 -370 {lab=vop}
N 420 -420 550 -420 {lab=von}
N 610 -420 750 -420 {lab=vop}
N 580 -400 580 -380 {lab=vdd}
N 700 -240 730 -240 {lab=vdd}
N 370 -230 400 -230 {lab=vdd}
C {symbols/nfet_03v3.sym} 440 -340 0 1 {name=M1
L=\{x_dut_m1_2_l\}
W=\{x_dut_m1_2_w\}
nf=2
m=1
ad="'int((nf+1)/2) * W/nf * 0.18u'"
pd="'2*int((nf+1)/2) * (W/nf + 0.18u)'"
as="'int((nf+2)/2) * W/nf * 0.18u'"
ps="'2*int((nf+2)/2) * (W/nf + 0.18u)'"
nrd="'0.18u / W'" nrs="'0.18u / W'"
sa=0 sb=0 sd=0
model=nfet_03v3
spiceprefix=X
}
C {symbols/nfet_03v3.sym} 730 -340 0 0 {name=M2
L=\{x_dut_m1_2_l\}
W=\{x_dut_m1_2_w\}
nf=2
m=1
ad="'int((nf+1)/2) * W/nf * 0.18u'"
pd="'2*int((nf+1)/2) * (W/nf + 0.18u)'"
as="'int((nf+2)/2) * W/nf * 0.18u'"
ps="'2*int((nf+2)/2) * (W/nf + 0.18u)'"
nrd="'0.18u / W'" nrs="'0.18u / W'"
sa=0 sb=0 sd=0
model=nfet_03v3
spiceprefix=X
}
C {devices/title.sym} 190 -50 0 0 {name=l1 author="Danial Noori Zadeh"}
C {devices/iopin.sym} 130 -640 0 0 {name=p1 lab=vdd}
C {devices/iopin.sym} 130 -600 0 0 {name=p2 lab=vss}
C {devices/ipin.sym} 360 -290 0 0 {name=p3 lab=inp}
C {devices/ipin.sym} 810 -290 0 1 {name=p4 lab=inn}
C {devices/opin.sym} 390 -420 0 1 {name=p5 lab=von}
C {devices/opin.sym} 780 -420 0 0 {name=p6 lab=vop}
C {devices/lab_pin.sym} 90 -640 0 0 {name=p11 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 90 -600 0 0 {name=p12 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 340 -340 0 0 {name=p13 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 840 -340 0 1 {name=p14 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 420 -180 0 0 {name=p15 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 750 -190 0 0 {name=p16 sig_type=std_logic lab=vss}
C {devices/ind.sym} 440 -510 0 0 {name=L2
m=1
value=\{x_dut_ldd\}
footprint=1206
device=inductor}
C {devices/ind.sym} 770 -510 0 0 {name=L3
m=1
value=\{x_dut_ldd\}
footprint=1206
device=inductor}
C {devices/lab_pin.sym} 400 -580 0 0 {name=p17 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 720 -580 0 0 {name=p18 sig_type=std_logic lab=vdd}
C {devices/ipin.sym} 580 -340 3 0 {name=p7 lab=vbias}
C {/foss/pdks/gf180mcuC/libs.tech/xschem//symbols/ppolyf_u_1k_6p0.sym} 420 -230 0 0 {name=R1
W=1e-6
L=1e-6
model=ppolyf_u_1k_6p0
spiceprefix=X
m=1}
C {/foss/pdks/gf180mcuC/libs.tech/xschem//symbols/ppolyf_u_1k_6p0.sym} 750 -240 0 0 {name=R2
W=1e-6
L=1e-6
model=ppolyf_u_1k_6p0
spiceprefix=X
m=1}
C {/foss/pdks/gf180mcuC/libs.tech/xschem//symbols/ppolyf_u_1k_6p0.sym} 580 -420 3 0 {name=R3
W=1e-6
L=1e-6
model=ppolyf_u_1k_6p0
spiceprefix=X
m=1}
C {devices/lab_pin.sym} 580 -380 0 0 {name=p8 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 370 -230 0 0 {name=p9 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 700 -240 0 0 {name=p10 sig_type=std_logic lab=vdd}
C {symbols/cap_mim_analog.sym} 380 -510 0 1 {name=C1
W=1e-6
L=1e-6
model=cap_mim_2f0_m3m4_noshield
spiceprefix=X
m=1}
C {symbols/cap_mim_analog.sym} 700 -510 0 1 {name=C2
W=1e-6
L=1e-6
model=cap_mim_2f0_m3m4_noshield
spiceprefix=X
m=1}
