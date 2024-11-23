from pathlib import Path
import hydra
import omegaconf
import pyrootutils
import pytest
import torch
import sys
sys.path.append("D:/ilib/hydra/lightning-hydra-template/")
from src.data.dataModule import DLIB300WDataModule


@pytest.mark.parametrize("batch_size", [32, 64])
def test_dlib300w_datamodule(batch_size):
    data_dir = "data/"

    dm = DLIB300WDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "ibug_300W_large_face_landmark_dataset").exists()
    assert Path(data_dir, "ibug_300W_large_face_landmark_dataset", "afw").exists()
    assert Path(data_dir, "ibug_300W_large_face_landmark_dataset", "lfpw").exists()
    assert Path(data_dir, "ibug_300W_large_face_landmark_dataset", "ibug").exists()
    assert Path(data_dir, "ibug_300W_large_face_landmark_dataset", "helen").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 6666

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    
    dm.draw_batch()