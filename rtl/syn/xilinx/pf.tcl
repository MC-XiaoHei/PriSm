set part @part@
read_verilog [list @rtl@/phaseformer_top.v]
read_xdc @syn@/pf.xdc
synth_design -top phaseformer_top -part $part -mode out_of_context
opt_design
place_design
route_design
report_utilization -file @out@/util.txt
report_timing_summary -file @out@/timing.txt
report_power -file @out@/power.txt
write_checkpoint -force @out@/routed.dcp
quit
