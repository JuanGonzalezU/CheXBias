python train_model.py \
--save female_0.pth \
--sex_proportion 100/0 \
--grouping sex \
--cuda_device 1 \
--experiment 1 ; /

python train_model.py \
--save female_20.pth \
--sex_proportion 80/20 \
--grouping sex \
--cuda_device 1 \
--experiment 1 ; /

python train_model.py \
--save female_40.pth \
--sex_proportion 60/40 \
--grouping sex \
--cuda_device 1 \
--experiment 1 ; /

python train_model.py \
--save female_60.pth \
--sex_proportion 40/60 \
--grouping sex \
--experiment 1 ; /

python train_model.py \
--save female_80.pth \
--sex_proportion 20/80 \
--grouping sex \
--cuda_device 1 \
--experiment 1 ; /
