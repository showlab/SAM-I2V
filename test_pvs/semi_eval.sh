################## SAM-I2V ##################

# setup prompt
#input="3c"
#input="bb"
input="gm"

# setup path
ckpt="sam-i2v_8gpu"
save_dir_name="sam-i2v_8gpu"

# setup workers
workers=64

# run evaluation
prediction_name="Semi_SAVTest_${ckpt}_${input}"
python ../tools/sav_evaluator.py \
--gt_root /workspace/i2v/data/sav_test/Annotations_6fps \
--pred_root ./output_semi/${save_dir_name}/${prediction_name} \
--num_processes ${workers}
