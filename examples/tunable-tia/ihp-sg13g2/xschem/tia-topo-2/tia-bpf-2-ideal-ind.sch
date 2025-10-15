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
N 840 -440 840 -430 {lab=vop}
N 840 -460 860 -460 {lab=vop}
N 500 -440 500 -430 {lab=von}
N 470 -460 500 -460 {lab=von}
N 440 -330 500 -330 {lab=inp}
N 500 -360 500 -330 {lab=inp}
N 840 -330 890 -330 {lab=inn}
N 840 -370 840 -330 {lab=inn}
N 840 -610 840 -580 {lab=vdd}
N 500 -610 500 -580 {lab=vdd}
N 840 -220 840 -160 {lab=vss}
N 500 -220 500 -160 {lab=vss}
N 350 -400 500 -400 {lab=vss}
N 760 -280 840 -280 {lab=inn}
N 840 -280 920 -280 {lab=inn}
N 840 -220 920 -220 {lab=vss}
N 760 -220 840 -220 {lab=vss}
N 420 -280 500 -280 {lab=inp}
N 500 -280 560 -280 {lab=inp}
N 500 -220 560 -220 {lab=vss}
N 420 -220 500 -220 {lab=vss}
N 840 -520 840 -460 {lab=vop}
N 300 -440 300 -430 {lab=von}
N 300 -440 500 -440 {lab=von}
N 500 -460 500 -440 {lab=von}
N 300 -370 300 -360 {lab=inp}
N 300 -360 500 -360 {lab=inp}
N 500 -370 500 -360 {lab=inp}
N 1100 -380 1100 -370 {lab=inn}
N 840 -370 1100 -370 {lab=inn}
N 840 -440 1100 -440 {lab=vop}
N 840 -460 840 -440 {lab=vop}
N 500 -520 500 -460 {lab=von}
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
C {sg13g2_pr/cap_cmim.sym} 420 -250 0 0 {name=C1
model=cap_cmim
w=\{x_dut_cap_w\}
l=\{x_dut_cap_l\}
m=\{x_dut_cap_m\}
spiceprefix=X}
C {sg13g2_pr/cap_cmim.sym} 920 -250 0 0 {name=C2
model=cap_cmim
w=\{x_dut_cap_w\}
l=\{x_dut_cap_l\}
m=\{x_dut_cap_m\}
spiceprefix=X}
C {sg13g2_pr/rppd.sym} 500 -550 0 1 {name=R1
w=\{x_dut_res_s_w\}
l=\{x_dut_res_s_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_s_m\}
}
C {sg13g2_pr/rppd.sym} 840 -550 0 0 {name=R2
w=\{x_dut_res_s_w\}
l=\{x_dut_res_s_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_s_m\}
}
C {sg13g2_pr/rppd.sym} 300 -400 0 1 {name=R3
w=\{x_dut_res_2_w\}
l=\{x_dut_res_2_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_2_m\}
}
C {ind.sym} 560 -250 0 0 {name=L2
m=1
value=\{x_dut_ind_size\}
footprint=1206
device=inductor}
C {ind.sym} 760 -250 0 0 {name=L3
m=1
value=\{x_dut_ind_size\}
footprint=1206
device=inductor}
C {devices/lab_pin.sym} 500 -610 0 0 {name=p13 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 840 -610 0 0 {name=p14 sig_type=std_logic lab=vdd}
C {devices/lab_pin.sym} 500 -160 0 0 {name=p16 sig_type=std_logic lab=vss}
C {devices/lab_pin.sym} 840 -160 0 1 {name=p17 sig_type=std_logic lab=vss}
C {sg13g2_pr/annotate_fet_params.sym} 3.75 -642.5 0 0 {name=annot1 ref=M1}
C {sg13g2_pr/rppd.sym} 1100 -410 0 0 {name=R4
w=\{x_dut_res_2_w\}
l=\{x_dut_res_2_l\}
model=rppd
spiceprefix=X
b=0
m=\{x_dut_res_2_m\}
}
