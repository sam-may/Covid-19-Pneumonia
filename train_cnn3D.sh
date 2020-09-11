python train_cnn3D.py \
--data_hdf5="/mnt/data/LungNodules/features_rotatable.hdf5" \
--metadata_json="/mnt/data/LungNodules/features_rotatable_metadata.json" \
--tag="nodulesCNN3D_v1-1-0" \
--max_epochs=100 \
--n_trainings=5 \
--training_batch_size=16 \
--validation_batch_size=16 \
--max_batch_size=64 \
--increase_batch \
--random_seed=0 \
--train_frac=0.7 \
--loss_function="weighted_crossentropy" \
--extra_features nodule_volume_norm \
--do_rotations

# Tag format: v-A-B-C
#     A --> extra features
#     None: 0, volume: 1, COM: 2, volume+COM: 3
#     B --> rotations
#     None: 0, azim: 1, polar: 2, azim+polar: 3
#     C --> nothing
