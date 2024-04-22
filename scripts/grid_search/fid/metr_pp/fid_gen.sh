for ((s = 60; s <= 100; s += 20 )); do
    for ((r = 4; r <= 58; r += 6 )); do
        accelerate launch -m metr.run_metr_fid\
          --project_name fid_msg_grid_srch_gen_no_vae \
          --run_name "r=$r s=$s" --w_channel 3 --w_pattern ring \
          --start 0 --end 500 \
          --with_tracking \
          --w_radius $r \
          --run_generation \
          --additional_metrics \
          --run_no_w \
          --image_folder /data/varlamov_a_data/tree-ring-watermark/msg_fid_grid_gen/r=$r_s=$s \
          --msg_type binary \
          --use_random_msgs \
          --msg_scaler $s \
          --target_clean_generated
    done
done1