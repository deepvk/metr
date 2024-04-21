accelerate launch -m tree_ring_watermark.run_tree_ring_watermark \
  --project_name gen_vanilla_tree \
  --run_name 1000_2000 --w_channel 3 --w_pattern ring \
  --start 1000 --end 2000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --save_locally \
  --local_path /data/varlamov_a_data/dima/imgs_vanilla_tree_start1000_end2000 
