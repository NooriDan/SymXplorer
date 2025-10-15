v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
N 210 -510 250 -510 {lab=vss}
N 210 -550 250 -550 {lab=vdd}
N 830 -450 970 -450 {lab=vbias}
N 830 -450 830 -430 {lab=vbias}
N 710 -450 830 -450 {lab=vbias}
N 1010 -450 1200 -450 {lab=vss}
N 1010 -510 1030 -510 {lab=vop}
N 640 -510 670 -510 {lab=von}
N 610 -380 670 -380 {lab=inp}
N 1010 -380 1060 -380 {lab=inn}
N 1010 -420 1010 -380 {lab=inn}
N 1010 -660 1010 -630 {lab=vdd}
N 670 -660 670 -630 {lab=vdd}
N 520 -450 670 -450 {lab=vss}
N 1010 -570 1010 -510 {lab=vop}
N 670 -510 670 -480 {lab=von}
N 670 -420 670 -380 {lab=inp}
N 1010 -510 1010 -480 {lab=vop}
N 670 -570 670 -510 {lab=von}
N 670 -630 710 -630 {lab=vdd}
N 620 -630 670 -630 {lab=vdd}
N 620 -570 670 -570 {lab=von}
N 670 -570 710 -570 {lab=von}
N 960 -630 1010 -630 {lab=vdd}
N 1010 -630 1070 -630 {lab=vdd}
N 1010 -570 1070 -570 {lab=vop}
N 960 -570 1010 -570 {lab=vop}
N 670 -380 670 -330 {lab=inp}
N 670 -270 670 -210 {lab=vss}
N 1010 -280 1010 -210 {lab=vss}
N 1010 -380 1010 -340 {lab=inn}
C {devices/title.sym} 350 -100 0 0 {name=l1 author="Danial Noori Zadeh"}
C {devices/iopin.sym} 250 -550 0 0 {name=p1 lab=vdd}
C {devices/iopin.sym} 250 -510 0 0 {name=p2 lab=vss}
C {devices/lab_pin.sym} 210 -550 0 0 {name=p11 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 210 -510 0 0 {name=p12 sig_type=std_logic lab=vss}
C {devices/ipin.sym} 610 -380 0 0 {name=p3 lab=inp}
C {devices/ipin.sym} 1060 -380 0 1 {name=p4 lab=inn}
C {devices/opin.sym} 640 -510 0 1 {name=p5 lab=von}
C {devices/opin.sym} 1030 -510 0 0 {name=p6 lab=vop}
C {devices/ipin.sym} 830 -430 3 0 {name=p7 lab=vbias}
C {sg13g2_pr/sg13_lv_rf_nmos.sym} 990 -450 0 0 {name=M1
l=\{x_dut_nfet_l\}
w=\{x_dut_nfet_w\}
ng=1
m=\{x_dut_nfet_m\}
rfmode=1
model=sg13_lv_nmos
spiceprefix=X
}
C {sg13g2_pr/sg13_lv_rf_nmos.sym} 690 -450 0 1 {name=M2
l=\{x_dut_nfet_l\}
w=\{x_dut_nfet_w\}
ng=1
m=\{x_dut_nfet_m\}
rfmode=1
model=sg13_lv_nmos
spiceprefix=X
}
C {devices/lab_pin.sym} 520 -450 0 0 {name=p8 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 1200 -450 0 1 {name=p9 sig_type=std_logic lab=vss}
C {ind.sym} 670 -300 0 1 {name=L2
m=1
value=\{x_dut_ind_size\}
footprint=1206
device=inductor}
C {ind.sym} 1010 -310 0 0 {name=L3
m=1
value=\{x_dut_ind_size\}
footprint=1206
device=inductor}
C {devices/lab_pin.sym} 670 -660 0 0 {name=p13 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 1010 -660 0 0 {name=p14 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 670 -210 0 0 {name=p16 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 1010 -210 0 1 {name=p17 sig_type=std_logic lab=vss}
C {sg13g2_pr/annotate_fet_params.sym} 173.75 -692.5 0 0 {name=annot1 ref=M1}
C {res.sym} 960 -600 0 1 {name=R1
value=\{x_dut_res_size\}
footprint=1206
device=resistor
m=1}
C {res.sym} 710 -600 0 0 {name=R2
value=\{x_dut_res_size\}
footprint=1206
device=resistor
m=1}
C {capa.sym} 1070 -600 0 0 {name=C1
m=1
value=\{x_dut_cap_size\}
footprint=1206
device="ceramic capacitor"}
C {capa.sym} 620 -600 0 1 {name=C2
m=1
value=\{x_dut_cap_size\}
footprint=1206
device="ceramic capacitor"}
