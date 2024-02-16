#----No tree-ring wm:-------

# AUTH_VAE_VANILLA_IMGS="/data/varlamov_a_data/dima/authors_stable_tree_3k_4k/imgs_no_w"

# ST_TREE_VAE_VANILLA_IMGS="/data/varlamov_a_data/dima/tree_tuned_3k_4k/imgs_no_w"

# accelerate launch -m tree_ring_watermark.eval_stable_tree \
#   --with_tracking \
#   --project_name eval_stable_tree \
#   --run_name vae_auth_imgs_no_tree \
#   --eval_imgs False --eval_bits True \
#   --img_dir $AUTH_VAE_VANILLA_IMGS \
#   --output_dir /data/varlamov_a_data/tree-ring-watermark/logs_vae_auth_imgs_st_tree \
#   --msg_decoder_path /data/varlamov_a_data/tree-ring-watermark/dec_48b_whit.torchscript.pt \
#   --attack_mode few

# accelerate launch -m tree_ring_watermark.eval_stable_tree \
  # --with_tracking \
  # --project_name eval_stable_tree \
  # --run_name fixed_vae_st_tr_imgs_no_tree \
  # --eval_imgs False --eval_bits True \
  # --img_dir $ST_TREE_VAE_VANILLA_IMGS \
  # --output_dir /data/varlamov_a_data/tree-ring-watermark/logs_vae_st_tr_imgs_st_tr \
  # --msg_decoder_path /data/varlamov_a_data/tree-ring-watermark/dec_48b_whit.torchscript.pt \
  # --attack_mode few

#----With tree-ring wm:-------

AUTH_VAE_STABLE_TREE_IMGS="/data/varlamov_a_data/dima/authors_stable_tree_3k_4k/imgs_w"

# ST_TREE_VAE_STABLE_TREE_IMGS="/data/varlamov_a_data/dima/tree_tuned_3k_4k/imgs_w"

accelerate launch -m tree_ring_watermark.eval_stable_tree \
  --with_tracking \
  --project_name eval_stable_tree \
  --run_name vae_auth_imgs_st_tree \
  --eval_imgs False --eval_bits True \
  --img_dir $AUTH_VAE_STABLE_TREE_IMGS \
  --output_dir /data/varlamov_a_data/tree-ring-watermark/logs_vae_auth_imgs_st_tree \
  --msg_decoder_path /data/varlamov_a_data/tree-ring-watermark/dec_48b_whit.torchscript.pt \
  --attack_mode few

# accelerate launch -m tree_ring_watermark.eval_stable_tree \
#   --with_tracking \
#   --project_name eval_stable_tree \
#   --run_name fixed_vae_st_tr_imgs_st_tr \
#   --eval_imgs False --eval_bits True \
#   --img_dir $ST_TREE_VAE_STABLE_TREE_IMGS \
#   --output_dir /data/varlamov_a_data/tree-ring-watermark/logs_vae_st_tr_imgs_st_tr \
#   --msg_decoder_path /data/varlamov_a_data/tree-ring-watermark/dec_48b_whit.torchscript.pt \
#   --attack_mode few