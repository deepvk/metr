for ((s = 60; s <= 140; s += 20 )); do
    for ((r = 4; r <= 58; r += 6 )); do
        accelerate launch -m metr.run_metr \
          --project_name msg_grid_srch \
          --run_name "r=$r s=$s" --w_channel 3 --w_pattern ring \
          --start 0 --end 100 \
          --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
          --with_tracking \
          --w_radius $r \
          --msg_type binary \
          --use_random_msgs \
          --msg_scaler $s 
    done
done