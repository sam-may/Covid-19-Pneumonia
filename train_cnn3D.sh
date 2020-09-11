python train_cnn3D.py \
--data_hdf5="/mnt/data/LungNodules/features_rotatable.hdf5" \
--metadata_json="/mnt/data/LungNodules/features_rotatable_metadata.json" \
# Tag format: v-A-B-C where A --> extra features, B --> rotations, C --> Nothing
--tag="nodulesCNN3D_v1-0-0" \
--max_epochs=100 \
--n_trainings=5 \
--training_batch_size=16 \
--validation_batch_size=16 \
--max_batch_size=64 \
--increase_batch \
--random_seed=0 \
--train_frac=0.7 \
--loss_function="weighted_crossentropy" \
# None: 0, volume: 1, COM: 2, volume+COM: 3
--extra_features nodule_volume_norm \
# None: 0, azim: 1, polar: 2, azim+polar: 3
# --do_rotations
