python train_unet.py \
--data_hdf5="/xilinx/scratch/covid_ct_data/14Jul2020_z_score_downsample256/features.hdf5" \
--metadata_json="/xilinx/scratch/covid_ct_data/14Jul2020_z_score_downsample256/features.json" \
--n_extra_slices=0 \
--extra_slice_step=1 \
--tag="2p5_0extra" \
--max_epochs=100 \
--n_trainings=1 \
--training_batch_size=16 \
--validation_batch_size=16 \
--max_batch_size=64 \
--random_seed=0 \
--train_frac=0.7 >& 2p5_0extra_logs.txt

python train_unet.py \
--data_hdf5="/xilinx/scratch/covid_ct_data/14Jul2020_z_score_downsample256/features.hdf5" \
--metadata_json="/xilinx/scratch/covid_ct_data/14Jul2020_z_score_downsample256/features.json" \
--n_extra_slices=3 \
--extra_slice_step=2 \
--tag="2p5_3extra_2step" \
--max_epochs=100 \
--n_trainings=1 \
--training_batch_size=16 \
--validation_batch_size=16 \
--max_batch_size=64 \
--random_seed=0 \
--train_frac=0.7 >& 2p5_3extra_2step_logs.txt

python train_unet.py \
--data_hdf5="/xilinx/scratch/covid_ct_data/14Jul2020_z_score_downsample256/features.hdf5" \
--metadata_json="/xilinx/scratch/covid_ct_data/14Jul2020_z_score_downsample256/features.json" \
--n_extra_slices=5 \
--extra_slice_step=1 \
--tag="2p5_5extra" \
--max_epochs=100 \
--n_trainings=1 \
--training_batch_size=16 \
--validation_batch_size=16 \
--max_batch_size=64 \
--random_seed=0 \
--train_frac=0.7 >& 2p5_5extra_logs.txt
