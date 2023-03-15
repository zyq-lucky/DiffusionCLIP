python main.py --edit_one_image            \
               --config celeba.yml         \
               --exp ./run_new/          \
               --t_0 150                    \
               --n_inv_step 5             \
               --n_test_step 10            \
               --n_iter 3                \
               --sample_type ddim          \
               --deterministic_inv 0        \
               --model_path  /home/zyq/.cache/torch/hub/checkpoints/celeba_hq.ckpt \
               --img_path $1  \

            #    --img_path /home/zyq/2022_new/main/GAN/CoCosNet/imgs/ade20k/training/ADE_train_00017763.jpg \
