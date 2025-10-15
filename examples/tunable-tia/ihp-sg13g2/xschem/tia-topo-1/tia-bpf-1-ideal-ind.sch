v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
N 40 -460 80 -460 {lab=vss}
N 40 -500 80 -500 {lab=vdd}
N 660 -400 800 -400 {lab=vbias}
N 660 -400 660 -380 {lab=vbias}
N 540 -400 660 -400 {lab=vbias}
N 840 -400 1030 -400 {lab=vss}
N 840 -330 840 -280 {lab=inn}
N 500 -330 500 -280 {lab=inp}
N 780 -520 780 -500 {lab=vop}
N 940 -520 940 -500 {lab=vop}
N 840 -460 840 -430 {lab=vop}
N 780 -500 840 -500 {lab=vop}
N 840 -460 860 -460 {lab=vop}
N 840 -470 840 -460 {lab=vop}
N 580 -520 580 -500 {lab=von}
N 500 -500 580 -500 {lab=von}
N 500 -460 500 -430 {lab=von}
N 470 -460 500 -460 {lab=von}
N 500 -470 500 -460 {lab=von}
N 520 -460 640 -460 {lab=von}
N 500 -470 520 -460 {lab=von}
N 500 -500 500 -470 {lab=von}
N 700 -460 820 -460 {lab=vop}
N 820 -460 840 -470 {lab=vop}
N 840 -500 840 -470 {lab=vop}
N 440 -330 500 -330 {lab=inp}
N 500 -370 500 -330 {lab=inp}
N 840 -330 890 -330 {lab=inn}
N 840 -370 840 -330 {lab=inn}
N 440 -610 440 -580 {lab=vdd}
N 940 -610 940 -580 {lab=vdd}
N 780 -610 780 -580 {lab=vdd}
N 580 -610 580 -580 {lab=vdd}
N 840 -220 840 -160 {lab=vss}
N 500 -220 500 -160 {lab=vss}
N 350 -400 500 -400 {lab=vss}
N 440 -520 440 -500 {lab=von}
N 840 -500 940 -500 {lab=vop}
N 440 -500 500 -500 {lab=von}
C {devices/title.sym} 180 -50 0 0 {name=l1 author="Danial Noori Zadeh"}
C {devices/iopin.sym} 80 -500 0 0 {name=p1 lab=vdd}
C {devices/iopin.sym} 80 -460 0 0 {name=p2 lab=vss}
C {devices/lab_pin.sym} 40 -500 0 0 {name=p11 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 40 -460 0 0 {name=p12 sig_type=std_logic lab=vss}
C {devices/ipin.sym} 440 -330 0 0 {name=p3 lab=inp}
C {devices/ipin.sym} 890 -330 0 1 {name=p4 lab=inn}
C {devices/opin.sym} 470 -460 0 1 {name=p5 lab=von}
C {devices/opin.sym} 860 -460 0 0 {name=p6 lab=vop}
C {devices/ipin.sym} 660 -380 3 0 {name=p7 lab=vbias}
C {sg13g2_pr/sg13_lv_rf_nmos.sym} 820 -400 0 0 {name=M1
l=\{x_dut_nfet_l\}
w=\{x_dut_nfet_w\}
ng=1
m=\{x_dut_nfet_m\}
rfmode=1
model=sg13_lv_nmos
spiceprefix=X
}
C {sg13g2_pr/sg13_lv_rf_nmos.sym} 520 -400 0 1 {name=M2
l=\{x_dut_nfet_l\}
w=\{x_dut_nfet_w\}
ng=1
m=\{x_dut_nfet_m\}
rfmode=1
model=sg13_lv_nmos
spiceprefix=X
}
C {devices/lab_pin.sym} 350 -400 0 0 {name=p8 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 1030 -400 0 1 {name=p9 sig_type=std_logic lab=vss}
C {sg13g2_pr/cap_cmim.sym} 440 -550 0 0 {name=C1
model=cap_cmim
w=\{x_dut_cap_w\}
l=\{x_dut_cap_l\}
m=\{x_dut_cap_m\}
spiceprefix=X}
C {sg13g2_pr/cap_cmim.sym} 940 -550 0 0 {name=C2
model=cap_cmim
w=\{x_dut_cap_w\}
l=\{x_dut_cap_l\}
m=\{x_dut_cap_m\}
spiceprefix=X}
C {sg13g2_pr/rppd.sym} 500 -250 0 1 {name=R1
w=\{x_dut_res_s_w\}
l=\{x_dut_res_s_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_s_m\}
}
C {sg13g2_pr/rppd.sym} 840 -250 0 0 {name=R2
w=\{x_dut_res_s_w\}
l=\{x_dut_res_s_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_s_m\}
}
C {sg13g2_pr/rppd.sym} 670 -460 3 0 {name=R3
w=\{x_dut_res_3_w\}
l=\{x_dut_res_3_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_3_m\}
}
C {ind.sym} 580 -550 0 0 {name=L2
m=1
value=\{x_dut_ind_size\}
footprint=1206
device=inductor}
C {ind.sym} 780 -550 0 0 {name=L3
m=1
value=\{x_dut_ind_size\}
footprint=1206
device=inductor}
C {devices/lab_pin.sym} 440 -610 0 0 {name=p10 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 580 -610 0 0 {name=p13 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 780 -610 0 0 {name=p14 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 940 -610 0 0 {name=p15 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 500 -160 0 0 {name=p16 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 840 -160 0 1 {name=p17 sig_type=std_logic lab=vss}
C {sg13g2_pr/annotate_fet_params.sym} 13.75 -412.5 0 0 {name=annot1 ref=M1}
