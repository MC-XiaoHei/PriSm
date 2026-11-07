set_device -name @family@ @part@
set_option -synthesis_tool gowinsynthesis
set_option -output_base_name tm
set_option -top_module timemixer_top
set_option -verilog_std sysv2017
set_option -place_option @place@
set_option -route_option @route@
add_file @syn@/tm.sdc
add_file @rtl@/timemixer_top.v
add_file @rtl@/dpram.v
run all
