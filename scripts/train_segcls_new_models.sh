export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=2b872565accf4439dd8fbdd0c2de356eb9c0214e

# python src/train.py experiment=seg_nodule_cls_lung_pos data.batch_size=32 trainer.devices=2 trainer.max_epochs=50
# python src/train.py experiment=seg_nodule_cls_nodule_lung_pos_lung_damage data.batch_size=4 trainer.devices=1 trainer.max_epochs=30
python src/train.py experiment=segcls_new_models data.batch_size=4 trainer.devices=1 trainer.max_epochs=30
