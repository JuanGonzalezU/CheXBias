python distribution.py \
--vae_path /home/juandres/aml/CheXBias/models/Experiment_2/sex/female_0/ \
--sex_proportion 100/0 ;

python distribution.py \
--vae_path /home/juandres/aml/CheXBias/models/Experiment_2/sex/female_20/ \
--sex_proportion 80/20 ;


python distribution.py \
--vae_path /home/juandres/aml/CheXBias/models/Experiment_2/sex/female_40/ \
--sex_proportion 60/40 ;


python distribution.py \
--vae_path /home/juandres/aml/CheXBias/models/Experiment_2/sex/female_60/ \
--sex_proportion 40/60 ;

python distribution.py \
--vae_path /home/juandres/aml/CheXBias/models/Experiment_2/sex/female_80/ \
--sex_proportion 20/80 ;

python distribution.py \
--vae_path /home/juandres/aml/CheXBias/models/Experiment_2/sex/female_100/ \
--sex_proportion 0/100 ;
