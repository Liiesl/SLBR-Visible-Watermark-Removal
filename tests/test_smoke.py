"""Smoke tests for the SLBR baseline on the current Python/PyTorch stack.

These tests intentionally run against the latest dependency versions. The
project's 2021 code is expected to fail some of them until the compatibility
and modernization work lands; each failure is a checklist item.

Imports are kept inside individual tests so one broken module cannot prevent
the remaining checks from running.
"""

import argparse
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MODULES = [
    "options",
    "evaluation",
    "datasets",
    "src.networks",
    "src.models",
    "src.utils.parallel",
    "src.utils.losses",
    "src.utils.misc",
    "pytorch_ssim",
    "pytorch_iou",
    "train",
    "test",
    "test_custom",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def _network_args(**overrides):
    args = SimpleNamespace(
        use_refine=True,
        mask_mode="res",
        sim_metric="cos",
        k_center=2,
        project_mode="simple",
        bg_mode="res_mask",
        k_refine=3,
        k_skip_stage=3,
        lr=1e-3,
        lambda_l1=4.0,
        lambda_mask=1.0,
        lambda_primary=0.01,
        lambda_content=0.0,
        lambda_style=0.0,
        lambda_iou=0.0,
        gan_norm=False,
        masked=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _build_network(**overrides):
    from src.networks import slbr

    torch.manual_seed(0)
    return slbr(args=_network_args(**overrides)).eval()


def _assert_output_contract(images, masks, watermark, size):
    assert isinstance(images, (list, tuple))
    assert isinstance(masks, (list, tuple))
    assert len(images) == 2
    assert len(masks) == 7
    assert watermark == [None]

    for image in images:
        assert image.shape == (1, 3, *size)
        assert torch.isfinite(image).all()
        assert image.min().item() >= 0.0
        assert image.max().item() <= 1.0

    assert masks[0].shape == (1, 1, *size)
    for mask in masks:
        assert torch.isfinite(mask).all()
        assert mask.min().item() >= 0.0
        assert mask.max().item() <= 1.0


def test_network_forward_square():
    model = _build_network()
    inputs = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        images, masks, watermark = model(inputs)

    _assert_output_contract(images, masks, watermark, (256, 256))
    assert masks[1].shape[-2:] == (64, 64)
    assert masks[3].shape[-2:] == (128, 128)
    assert masks[5].shape[-2:] == (256, 256)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SLBR is square-only until the SLBR-Next shape fix; the refinement "
        "stage interpolates with reversed H/W."
    ),
)
def test_network_forward_non_square():
    model = _build_network()
    inputs = torch.rand(1, 3, 192, 256)

    with torch.no_grad():
        model(inputs)


def test_training_step_through_model_factory(tmp_path):
    from options import Options
    import src.models as models

    parser = Options().init(argparse.ArgumentParser())
    args = parser.parse_args([])
    args.nets = "slbr"
    args.models = "slbr"
    args.name = "ci"
    args.checkpoint = str(tmp_path)
    args.evaluate = True
    args.use_refine = True
    args.mask_mode = "res"
    args.k_center = 2
    args.k_refine = 3
    args.k_skip_stage = 3
    args.lambda_l1 = 4.0
    args.lambda_mask = 1.0
    args.lambda_primary = 0.01
    args.lambda_content = 0.0
    args.lambda_style = 0.0
    args.lambda_iou = 0.0

    machine = models.slbr(datasets=(None, None), args=args)
    model = machine.model
    model.train()
    model.zero_grad_all()

    inputs = torch.rand(2, 3, 128, 128, device=machine.device)
    target = torch.rand_like(inputs)
    mask = (torch.rand(2, 1, 128, 128, device=machine.device) > 0.5).float()

    outputs = model(machine.norm(inputs))
    coarse_loss, refine_loss, vgg_loss, mask_loss = machine.loss(
        inputs, outputs[0], machine.norm(target), outputs[1], mask
    )
    total_loss = (
        args.lambda_l1 * (coarse_loss + refine_loss)
        + args.lambda_mask * mask_loss
        + vgg_loss
    )
    total_loss.backward()
    model.step_all()

    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)


@pytest.mark.parametrize("preprocess", ["resize", "resize_and_crop"])
def test_dataset_transform_chain(preprocess):
    from datasets.base_dataset import get_transform

    args = SimpleNamespace(
        preprocess=preprocess,
        input_size=256,
        crop_size=256,
        no_flip=False,
    )
    transform = get_transform(args, additional_targets={"mask": "mask"})

    image = np.zeros((300, 300, 3), dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=np.uint8)
    output = transform(image=image, mask=mask)

    assert output["image"].shape == (256, 256, 3)
    assert output["mask"].shape == (256, 256)


def test_fmeasure_smoke():
    from evaluation import Fmeasure

    fm = Fmeasure(1)
    assert fm.n_imgs == 1


def test_custom_preprocess(tmp_path):
    import cv2
    from test_custom import preprocess

    image_path = tmp_path / "sample.png"
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    image[..., 1] = 127
    assert cv2.imwrite(str(image_path), image)

    tensor = preprocess(str(image_path), img_size=64)
    assert tensor.shape == (1, 3, 64, 64)
    assert torch.isfinite(tensor).all()
