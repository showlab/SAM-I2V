################## SAM-I2V ##################

# setup prompt
#input="3c"
#input="bb"
input="gm"

# setup path
ckpt="sam-i2v_8gpu"
yaml="i2v-infer.yaml"
save_dir_name="sam-i2v_8gpu"

# run inference
python inference_Semi_SAV_mgpu.py \
--input ${input} \
--ckpt ${ckpt} \
--yaml ${yaml} \
--save_dir_name ${save_dir_name}
