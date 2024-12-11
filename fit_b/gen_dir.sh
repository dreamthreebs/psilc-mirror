old_fold="./270GHz"
new_fold="./155GHz"
mkdir -p ./$new_fold

mkdir -p ./$new_fold/inpainting
mkdir -p ./$new_fold/fit_res
mkdir -p ./$new_fold/mask

cp ./$old_fold/* ./$new_fold

# cp ./$old_fold/inpainting/* ./$new_fold/inpainting
# cp ./$old_fold/fit_res/* ./$new_fold/fit_res
# cp ./$old_fold/mask/* ./$new_fold/mask



