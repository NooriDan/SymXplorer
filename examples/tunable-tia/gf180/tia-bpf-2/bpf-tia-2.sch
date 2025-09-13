v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
N 740 -420 770 -420 {lab=vop}
N 380 -420 410 -420 {lab=von}
N 350 -290 410 -290 {lab=inp}
N 740 -290 800 -290 {lab=inn}
N 80 -600 120 -600 {lab=vss}
N 80 -640 120 -640 {lab=vdd}
N 450 -340 700 -340 {lab=vbias}
N 330 -340 410 -340 {lab=vss}
N 740 -340 830 -340 {lab=vss}
N 410 -310 410 -260 {lab=inp}
N 740 -310 740 -270 {lab=inn}
N 410 -200 410 -180 {lab=vss}
N 740 -210 740 -190 {lab=vss}
N 430 -480 430 -460 {lab=von}
N 370 -460 430 -460 {lab=von}
N 370 -480 370 -460 {lab=von}
N 370 -550 370 -540 {lab=vdd}
N 370 -550 430 -550 {lab=vdd}
N 430 -550 430 -540 {lab=vdd}
N 390 -580 390 -550 {lab=vdd}
N 410 -460 410 -370 {lab=von}
N 690 -550 690 -540 {lab=vdd}
N 690 -550 760 -550 {lab=vdd}
N 760 -550 760 -540 {lab=vdd}
N 710 -580 710 -550 {lab=vdd}
N 690 -480 690 -460 {lab=vop}
N 690 -460 760 -460 {lab=vop}
N 760 -480 760 -460 {lab=vop}
N 740 -460 740 -370 {lab=vop}
N 470 -510 650 -510 {lab=VR}
N 430 -540 430 -510 {lab=vdd}
N 690 -540 690 -510 {lab=vdd}
C {symbols/nfet_03v3.sym} 430 -340 0 1 {name=M1
L=\{x_dut_m1_2_l\}
W=\{x_dut_m1_2_w\}
nf=1
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
C {symbols/nfet_03v3.sym} 720 -340 0 0 {name=M2
L=\{x_dut_m1_2_l\}
W=\{x_dut_m1_2_w\}
nf=1
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
C {devices/title.sym} 180 -50 0 0 {name=l1 author="Danial Noori Zadeh"}
C {devices/iopin.sym} 120 -640 0 0 {name=p1 lab=vdd}
C {devices/iopin.sym} 120 -600 0 0 {name=p2 lab=vss}
C {devices/ipin.sym} 350 -290 0 0 {name=p3 lab=inp}
C {devices/ipin.sym} 800 -290 0 1 {name=p4 lab=inn}
C {devices/opin.sym} 380 -420 0 1 {name=p5 lab=von}
C {devices/opin.sym} 770 -420 0 0 {name=p6 lab=vop}
C {devices/lab_pin.sym} 80 -640 0 0 {name=p11 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 80 -600 0 0 {name=p12 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 330 -340 0 0 {name=p13 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 830 -340 0 1 {name=p14 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 410 -180 0 0 {name=p15 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 740 -190 0 0 {name=p16 sig_type=std_logic lab=vss}
C {devices/capa.sym} 370 -510 0 0 {name=C1
m=1
value=\{x_dut_cdd\}
footprint=1206
device="ceramic capacitor"}
C {devices/capa.sym} 760 -510 0 0 {name=C2
m=1
value=\{x_dut_cdd\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_pin.sym} 390 -580 0 0 {name=p17 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 710 -580 0 0 {name=p18 sig_type=std_logic lab=vdd}
C {devices/ipin.sym} 570 -340 3 0 {name=p7 lab=vbias}
C {symbols/pfet_03v3.sym} 450 -510 0 1 {name=M3
L=\{x_dut_m3_4_l\}
W=\{x_dut_m3_4_w\}
nf=1
m=1
ad="'int((nf+1)/2) * W/nf * 0.18u'"
pd="'2*int((nf+1)/2) * (W/nf + 0.18u)'"
as="'int((nf+2)/2) * W/nf * 0.18u'"
ps="'2*int((nf+2)/2) * (W/nf + 0.18u)'"
nrd="'0.18u / W'" nrs="'0.18u / W'"
sa=0 sb=0 sd=0
model=pfet_03v3
spiceprefix=X
}
C {symbols/pfet_03v3.sym} 670 -510 0 0 {name=M4
L=\{x_dut_m3_4_l\}
W=\{x_dut_m3_4_w\}
nf=1
m=1
ad="'int((nf+1)/2) * W/nf * 0.18u'"
pd="'2*int((nf+1)/2) * (W/nf + 0.18u)'"
as="'int((nf+2)/2) * W/nf * 0.18u'"
ps="'2*int((nf+2)/2) * (W/nf + 0.18u)'"
nrd="'0.18u / W'" nrs="'0.18u / W'"
sa=0 sb=0 sd=0
model=pfet_03v3
spiceprefix=X
}
C {devices/ipin.sym} 560 -510 3 0 {name=p8 lab=VR}
C {devices/ind.sym} 410 -230 0 0 {name=L2
m=1
value=\{x_dut_lss\}
footprint=1206
device=inductor}
C {devices/ind.sym} 740 -240 0 0 {name=L3
m=1
value=\{x_dut_lss\}
footprint=1206
device=inductor}
