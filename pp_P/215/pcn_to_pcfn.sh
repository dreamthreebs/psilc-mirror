run_pcn_file="./run_pcn.py"
run_pcfn_file="./run_pcfn.py"
cp $run_pcn_file $run_pcfn_file
sed -i "s/PSCMBNOISE/PSCMBFGNOISE/g" $run_pcfn_file

submit_pcn_file="./submit_pcn.sh"
submit_pcfn_file="./submit_pcfn.sh"
cp $submit_pcn_file $submit_pcfn_file
sed -i "s/run_pcn/run_pcfn/g" $submit_pcfn_file

gen_pcn_file="./gen_run_pcn.sh"
gen_pcfn_file="./gen_run_pcfn.sh"
cp $gen_pcn_file $gen_pcfn_file
sed -i "s/pcn/pcfn/g" $gen_pcfn_file

gen_submit_pcn_file="./gen_submit_pcn.sh"
gen_submit_pcfn_file="./gen_submit_pcfn.sh"
cp $gen_submit_pcn_file $gen_submit_pcfn_file
sed -i "s/pcn/pcfn/g" $gen_submit_pcfn_file

do_pcn_file="./do_pcn.sh"
do_pcfn_file="./do_pcfn.sh"
cp $do_pcn_file $do_pcfn_file
sed -i "s/pcn/pcfn/g" $do_pcfn_file
