set_device -name @family@ @part@
set_option -synthesis_tool gowinsynthesis
set_option -output_base_name prism
set_option -top_module prism_top
set_option -verilog_std sysv2017
set_option -place_option @place@
add_file @syn@/prism.sdc
add_file @rtl@/prism_top.v
add_file @rtl@/lattice_w.v
add_file @rtl@/dpram.v
run all
