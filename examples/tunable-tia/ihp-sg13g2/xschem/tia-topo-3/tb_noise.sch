v {xschem version=3.4.7 file_version=1.2}
G {}
K {}
V {}
S {}
E {}
L 4 100 -950 410 -950 {}
L 4 410 -950 410 -730 {}
L 4 90 -730 410 -730 {}
L 4 90 -950 90 -730 {}
L 4 90 -950 100 -950 {}
L 4 90 -610 670 -610 {}
L 4 670 -610 670 -370 {}
L 4 90 -370 670 -370 {}
L 4 90 -610 90 -370 {}
L 4 450 -950 1190 -950 {}
L 4 1190 -950 1190 -650 {}
L 4 450 -650 1190 -650 {}
L 4 450 -950 450 -650 {}
L 4 712.5 -610 1010 -610 {}
L 4 1010 -610 1010 -376.25 {}
L 4 710 -610 710 -377.5 {}
L 4 710 -610 712.5 -610 {}
L 4 1010 -376.25 1010 -370 {}
L 4 710 -370 1010 -370 {}
L 4 710 -377.5 710 -371.25 {}
L 4 710 -371.25 710 -370 {}
T {Supply} 100 -970 0 0 0.2 0.2 {}
T {Stimulus} 100 -630 0 0 0.2 0.2 {}
T {DUT} 460 -970 0 0 0.2 0.2 {}
T {Load} 715 -625 0 0 0.2 0.2 {}
N 170 -810 170 -790 {
lab=GND}
N 170 -910 170 -870 {
lab=VSS}
N 250 -810 250 -790 {
lab=GND}
N 170 -790 170 -770 {
lab=GND}
N 250 -910 250 -870 {
lab=VDD}
N 170 -790 250 -790 {
lab=GND}
N 340 -440 340 -420 {
lab=GND}
N 430 -440 430 -420 {
lab=GND}
N 940 -860 980 -860 {lab=VDD}
N 940 -840 980 -840 {lab=VSS}
N 330 -810 330 -790 {lab=GND}
N 250 -790 330 -790 {lab=GND}
N 330 -910 330 -870 {lab=Vbias}
N 938.75 -810 978.75 -810 {lab=Vop}
N 938.75 -790 978.75 -790 {lab=Von}
N 750 -466.25 750 -456.25 {lab=GND}
N 750 -466.25 830 -466.25 {lab=GND}
N 830 -476.25 830 -466.25 {lab=GND}
N 830 -556.25 830 -536.25 {lab=Vop}
N 750 -556.25 750 -536.25 {lab=Von}
N 340 -520 340 -500 {lab=In}
N 390 -520 430 -520 {lab=In}
N 430 -520 430 -500 {lab=In}
N 390 -580 390 -520 {lab=In}
N 160 -440 160 -420 {
lab=GND}
N 250 -440 250 -420 {
lab=GND}
N 160 -520 160 -500 {lab=Ip}
N 210 -520 250 -520 {lab=Ip}
N 250 -520 250 -500 {lab=Ip}
N 210 -580 210 -520 {lab=Ip}
N 750 -476.25 750 -466.25 {lab=GND}
N 340 -520 390 -520 {lab=In}
N 160 -520 210 -520 {lab=Ip}
N 570 -871.25 570 -850 {lab=Vbias}
N 550 -871.25 570 -871.25 {lab=Vbias}
N 570 -810 570 -790 {lab=Ip}
N 550 -790 570 -790 {lab=Ip}
N 532.5 -830 642.5 -830 {lab=In}
N 570 -850 642.5 -850 {lab=Vbias}
N 570 -810 642.5 -810 {lab=Ip}
C {devices/title.sym} 160 -30 0 0 {name=l5 author="(c) 2025-2026 D. Noori Zadeh, Apache-2.0 license"}
C {devices/spice_probe.sym} 540 -830 0 0 {name=p5 attrs=""}
C {devices/code.sym} 544.716979577353 -240 0 0 {name=controls
simulator=ngspice
only_toplevel=false
value="
****** START OF CONTROL SECTION ****** 
.temp 27
.control
    option sparse
    set filetype=ascii
    save all

    * --- AC Analysis ---
    * Optional for debug
    * ac dec 101 1k 1G
    * plot 20*log10(v(Vop) - v(Von))
    * echo AC Analysis Complete.

    echo =============================================
    echo Starting Noise Analysis...
    echo     (Output: v(Vop, Von), Input: Idp1)
    echo =============================================

    * --- Noise Analysis ---
    * [EDIT] will need to change the frequency limits as needed.
    noise v(Vop, Von) Idp1 lin 1000 1MEG 1000MEG 1
    echo Noise Analysis Complete.

    * Set the plot type to noise1 since this is where the information about noise densities are stored
    setplot noise1
    plot inoise_spectrum

    echo ---------------------------------------------
    echo Calculating Total Integrated Noise Metrics...
    echo ---------------------------------------------
    * Set the plot type to noise2 since this is where the integerated values are stored.
    setplot noise2

    * --- Save Metrics as Variables ---
    * Create new vectors in the 'noise2' plot (which is now active)
    let total_input_noise_sqr = inoise_total
    let total_output_noise_sqr = onoise_total
    let total_input_noise_rms = sqrt(inoise_total)
    let total_output_noise_rms = sqrt(onoise_total)

    * --- Print Metrics to Log ---
    * Use echo and print to show the final RMS values clearly
    echo
    echo Total Input-Referred Noise (RMS):
    print total_input_noise_rms
    echo
    echo Total Output-Referred Noise (RMS):
    print total_output_noise_rms
    echo

    * --- Write to File ---
    echo Saving noise spectra and metrics to rawspice.raw (has to have this name)...
    * Only save whats needed.
    write rawspice.raw noise1.inoise_spectrum noise1.onoise_spectrum noise2.inoise_total noise2.onoise_total
    
    echo Save Complete.
    echo =============================================
    echo Simulation Finished.
    echo =============================================
    * quit

.endc

******  END OF CONTROL SECTION ****** 
"}
C {devices/code.sym} 60 -250 0 0 {name=MODELS only_toplevel=true
format="tcleval( @value )"
value="
********** Models ************
.lib cornerMOSlv.lib mos_tt
.lib cornerRES.lib res_typ 
.lib $::SG13G2_MODELS/cornerCAP.lib cap_typ
********** END of Models ************
"}
C {devices/code.sym} 417.5 -250 0 0 {name=save_ngspice only_toplevel=false value="

****** Save Statements ****** 
*.option savecurrents

** nfet small signal parameters **
*.save @n.x_dut.xm1.nsg13_lv_nmos[cdd]
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
C {devices/code.sym} 300 -250 0 0 {name=tb_params only_toplevel=false value="
****** TB Parameters ****** 
.param icm=0.5e-6
.param vdd=1.2
.param vbias=1
.param cload=100p
******  End of TB Params ****** 
"}
C {devices/code.sym} 180 -250 0 0 {name=dut_params only_toplevel=false value="
****** All DUT parameters ******

* MOSFET PARAMETERS
.param x_dut_nfet_w=0.40u
.param x_dut_nfet_l=0.26u
.param x_dut_nfet_m=10

* PASSIVE ELEMENT PARAMETERS

.param x_dut_cap_size=500f
.param x_dut_res_size=100.0k
.param x_dut_ind_size=200.0n

****** END of DUT Params ****** 
"}
C {devices/lab_wire.sym} 980 -860 0 1 {name=p10 sig_type=std_logic lab=VDD}
C {devices/launcher.sym} 1018.75 -261.25 0 0 {name=h1
descr="Annotate OP" 
tclcommand="
set show_hidden_texts 1; 
xschem annotate_op
"
}
C {devices/launcher.sym} 1020 -200 0 0 {name=h2
descr="Load waves AC" 
tclcommand="
xschem raw_read $netlist_dir/rawspice.raw ac

"
}
C {devices/vsource.sym} 170 -840 0 0 {name=V0 value=0 savecurrent=false}
C {devices/gnd.sym} 170 -770 0 0 {name=l9 lab=GND}
C {devices/vsource.sym} 250 -840 0 0 {name=V2 value=\{vdd\} savecurrent=false}
C {devices/lab_wire.sym} 170 -910 0 0 {name=p15 sig_type=std_logic lab=VSS}
C {devices/gnd.sym} 340 -420 0 0 {name=l10 lab=GND}
C {devices/gnd.sym} 430 -420 0 0 {name=l11 lab=GND}
C {devices/lab_wire.sym} 390 -580 0 0 {name=p16 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 250 -910 0 0 {name=p17 sig_type=std_logic lab=VDD}
C {devices/lab_wire.sym} 980 -840 0 1 {name=p18 sig_type=std_logic lab=VSS}
C {devices/vsource.sym} 330 -840 0 0 {name=V1 value=\{vbias\} savecurrent=false}
C {devices/lab_wire.sym} 330 -910 0 0 {name=p19 sig_type=std_logic lab=Vbias}
C {devices/lab_wire.sym} 550 -871.25 0 0 {name=p20 sig_type=std_logic lab=Vbias}
C {devices/capa.sym} 750 -506.25 0 0 {name=C4
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/lab_wire.sym} 978.75 -810 0 1 {name=p21 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 978.75 -790 0 1 {name=p22 sig_type=std_logic lab=Von}
C {devices/lab_wire.sym} 532.5 -830 0 0 {name=p23 sig_type=std_logic lab=In}
C {devices/lab_wire.sym} 550 -790 0 0 {name=p24 sig_type=std_logic lab=Ip}
C {devices/capa.sym} 830 -506.25 0 0 {name=C5
m=1
value=\{cload\}
footprint=1206
device="ceramic capacitor"}
C {devices/gnd.sym} 750 -456.25 0 0 {name=l12 lab=GND}
C {devices/lab_wire.sym} 830 -556.25 0 1 {name=p25 sig_type=std_logic lab=Vop}
C {devices/lab_wire.sym} 750 -556.25 0 1 {name=p26 sig_type=std_logic lab=Von}
C {devices/isource.sym} 430 -470 0 0 {name=Icm2 value=\{icm\}}
C {devices/isource.sym} 340 -470 0 0 {name=Idn1 value="dc 0 ac -0.5"}
C {devices/gnd.sym} 160 -420 0 0 {name=l13 lab=GND}
C {devices/gnd.sym} 250 -420 0 0 {name=l14 lab=GND}
C {devices/lab_wire.sym} 210 -580 0 0 {name=p27 sig_type=std_logic lab=Ip}
C {devices/isource.sym} 250 -470 0 0 {name=Icm3 value=\{icm\}}
C {devices/isource.sym} 160 -470 0 0 {name=Idp1 value="dc 0 ac 0.5"}
C {tunable-tia/ihp-sg13g2/xschem/tia-topo-3/tia-bpf-3-ideal-component.sym} 642.5 -760 0 0 {name=x_dut1 }
C {devices/spice_probe.sym} 550 -790 0 0 {name=p1 attrs=""}
C {devices/spice_probe.sym} 960 -810 0 0 {name=p2 attrs=""}
C {devices/spice_probe.sym} 960 -790 0 0 {name=p3 attrs=""}
