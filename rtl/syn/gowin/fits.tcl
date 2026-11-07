set_device -name @family@ @part@
set_option -synthesis_tool gowinsynthesis
set_option -output_base_name fits
set_option -top_module fits_top
set_option -verilog_std sysv2017
set_option -place_option @place@
add_file @syn@/fits.sdc
add_file @rtl@/fits_top.v
run all
