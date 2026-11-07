set part @part@
read_verilog [list @rtl@/timekan_top.v]
read_xdc @syn@/tk.xdc
synth_design -top timekan_top -part $part -mode out_of_context
opt_design
place_design
route_design
report_utilization -file @out@/util.txt
report_timing_summary -file @out@/timing.txt
report_power -file @out@/power.txt
write_checkpoint -force @out@/routed.dcp
quit
