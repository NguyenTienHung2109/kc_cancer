export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=2b872565accf4439dd8fbdd0c2de356eb9c0214e
python src/train.py experiment=lidc_segcls data.batch_size=16 trainer.devices=1

# cd kc_cancer
# conda activate ./env
# cd new_models
