names=(
    "jpeg"
    "crop"
    "blur"
    "noise"
    "brightness"
    "rotate"
    "diff_150"
    "no_attack"
    "vae_2018_q_1"
)

PROJECT=generate_stable_tree_all_attacks
OUTPUT_ROOT=all_attacks/stable_sig

for ((i=0; i<${#names[@]}; i++)); do
    accelerate launch -m metr.metr_pp_eval_stable_sig \
      --with_tracking \
      --project_name $PROJECT \
      --run_name ${names[i]} \
      --eval_imgs False --eval_bits True \
      --img_dir $OUTPUT_ROOT/${names[i]}/imgs_w \
      --output_dir all_attacks_logs \
      --msg_decoder_path dec_48b_whit.torchscript.pt \
      --attack_mode none \
      --key_str 111010110101000001010111010011010100010000100111
done