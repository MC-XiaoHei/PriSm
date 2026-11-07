set_device -name @family@ @part@
set_option -synthesis_tool gowinsynthesis
set_option -output_base_name tk
set_option -top_module timekan_top
set_option -verilog_std sysv2017
set_option -place_option @place@
set_option -route_option @route@
add_file @syn@/tk.sdc
add_file @rtl@/timekan_top.v
run all
