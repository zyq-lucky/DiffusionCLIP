# for name in /home/zyq/2022_new/main/ESPM/results/celebahq/single/*.jpg ; 
# # do printf '%s is a directory\n' "$name"; 
# do bash edit_one_image.sh $name
# done
python main.py --edit_images_from_dataset  \
               --config celeba.yml         \
               --exp ./output_test/all/           \
               --n_test_img 50             \
               --t_0 500                   \
               --n_inv_step 40             \
               --n_test_step 20            \
               --n_iter 3                \
               --deterministic_inv 0        \
               --sample_type ddim          \
               --bs_train 1                \
               --model_path /home/zyq/.cache/torch/hub/checkpoints/celeba_hq.ckpt 