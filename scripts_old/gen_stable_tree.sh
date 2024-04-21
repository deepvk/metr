accelerate launch -m tree_ring_watermark.run_stable_tree \
  --project_name gen_stable_tree \
  --run_name auth_3k_4k --w_channel 3 --w_pattern ring \
  --start 3000 --end 4000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --save_locally \
  --local_path /data/varlamov_a_data/dima/authors_stable_tree_3k_4k \
  --decoder_state_dict_path /data/varlamov_a_data/tree-ring-watermark/ldm_decoders/authors.pth