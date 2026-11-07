set_device -name @family@ @part@
set_option -synthesis_tool gowinsynthesis
set_option -output_base_name pf
set_option -top_module phaseformer_top
set_option -verilog_std sysv2017
set_option -place_option @place@
add_file @syn@/pf.sdc
add_file @rtl@/phaseformer_top.v
run all
