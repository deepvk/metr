S=(
    60
    70
    80
    90
    100
)

messages=(
  "1111111111"
  "0000000000"
  "1010101010"
  "1100110011"
  "0011001100"
  "1111100000"
  "0000011111"
)

for ((j=0; j<${#messages[@]}; j++)); do
    for ((i=0; i<${#S[@]}; i++)); do
        accelerate launch -m metr.run_metr_fid \
            --project_name fid_gеn_message_s \
            --run_name ${S[i]}_${messages[j]} --w_channel 3 --w_pattern ring \
            --start 0 --end 5000 \
            --with_tracking \
            --w_radius 10 \
            --run_generation \
            --additional_metrics \
            --run_no_w \
            --image_folder worst_message/gen_${S[i]}_${messages[j]}  \
            --msg_type binary \
            --msg ${messages[j]} \
            --msg_scaler ${S[i]} \
            --target_clean_generated
    done
done