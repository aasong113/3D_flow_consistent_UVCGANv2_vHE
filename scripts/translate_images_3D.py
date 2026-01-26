#!/usr/bin/env python

import argparse
import collections
import os
import sys

import tqdm
import numpy as np
from PIL import Image
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from uvcgan2.consts import MERGE_NONE
from uvcgan2.eval.funcs import (
    start_model_eval, tensor_to_image, slice_data_loader, get_eval_savedir,
    make_image_subdirs
)
from uvcgan2.data import construct_data_loaders
from uvcgan2.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions as images'
    )
    add_standard_eval_parsers(parser, default_epoch = 10)
    add_plot_extension_parser(parser)

    parser.add_argument(
        '--data-root',
        '--dataroot',
        default = None,
        dest    = 'data_root',
        help    = (
            'override dataset root. For CycleGAN-style datasets this should be '
            'the directory containing trainA/trainB/testA/testB; if a split '
            'directory (e.g. testA) is provided, its parent is used.'
        ),
        type    = str,
    )

    return parser.parse_args()

def _infer_cyclegan_root(path):
    path = os.path.abspath(os.path.expanduser(path))
    base = os.path.basename(path).lower()

    split_dirs = {'traina', 'trainb', 'testa', 'testb', 'vala', 'valb'}
    if base in split_dirs:
        return os.path.dirname(path)

    return path

def _infer_domain_from_split_dir(path):
    base = os.path.basename(os.path.abspath(os.path.expanduser(path))).lower()
    split_prefixes = ('train', 'test', 'val')
    if not any(base.startswith(p) for p in split_prefixes):
        return None

    if base.endswith('a'):
        return 'a'
    if base.endswith('b'):
        return 'b'

    return None

def _override_dataset_paths(data_config, data_root):
    if not data_root:
        return

    inferred_root = _infer_cyclegan_root(data_root)

    for ds_conf in data_config.datasets:
        if isinstance(ds_conf.dataset, dict):
            ds_conf.dataset['path'] = inferred_root

def _coerce_domain_a_adjacent_z_pairs_to_cyclegan(data_config):
    for ds_conf in data_config.datasets:
        if not isinstance(ds_conf.dataset, dict):
            continue

        if ds_conf.dataset.get('name') != 'adjacent-z-pairs':
            continue

        domain = str(ds_conf.dataset.get('domain', '')).lower()
        if domain != 'a':
            continue

        kept = {}
        for key in ('domain', 'path', 'inference'):
            if key in ds_conf.dataset:
                kept[key] = ds_conf.dataset[key]

        kept['name'] = 'cyclegan'
        ds_conf.dataset = kept

def _restrict_to_domain(data_config, domain):
    domain = str(domain).lower()
    kept = []
    for ds_conf in data_config.datasets:
        if not isinstance(ds_conf.dataset, dict):
            kept.append(ds_conf)
            continue

        ds_domain = str(ds_conf.dataset.get('domain', '')).lower()
        if ds_domain == domain:
            kept.append(ds_conf)

    data_config.datasets = kept

def _drop_missing_cyclegan_domains(data_config, split):
    kept = []
    for ds_conf in data_config.datasets:
        if not isinstance(ds_conf.dataset, dict):
            kept.append(ds_conf)
            continue

        name = ds_conf.dataset.get('name')
        if name not in {'cyclegan', 'image-domain-folder'}:
            kept.append(ds_conf)
            continue

        root = ds_conf.dataset.get('path')
        domain = ds_conf.dataset.get('domain', 'a')
        if root is None:
            kept.append(ds_conf)
            continue

        expected = os.path.join(str(root), f"{split}{str(domain).upper()}")
        if os.path.isdir(expected):
            kept.append(ds_conf)

    data_config.datasets = kept

def save_images(model, savedir, filenames, ext):
    """Save model outputs using original filenames."""
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        # model.images[name] is a batch of outputs: shape (N, C, H, W)
        for idx in range(torch_image.shape[0]):

            # ---- original filename corresponding to this output ----
            original_name = filenames[idx]

            # remove original extension, add new one later
            base = os.path.splitext(original_name)[0]

            # convert tensor → numpy uint8 image
            image = tensor_to_image(torch_image[idx])
            image = np.round(255 * image).astype(np.uint8)
            image = Image.fromarray(image)

            for e in ext:
                out_path = os.path.join(savedir, name, f"{base}.{e}")
                image.save(out_path)

def dump_single_domain_images(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter, ext
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
    desc = f'Translating domain {domain}'

    for batch in tqdm.tqdm(data_it, desc = desc, total = steps):
        #print(batch)
        # batch = [(tensor, filename), (tensor, filename), ...]
        #print(batch)
        images, names = batch

        # Convert list of tensors → batch tensor (N,C,H,W)
        images = torch.stack(images, dim=0)

        model.set_input(images, domain=domain)

        # and store filenames in model or return them later
        model.filenames = names

        torch.autograd.set_detect_anomaly(True)
        model.forward_nograd()

        save_images(model, savedir, names, ext)

def dump_images(model, data_list, n_eval, batch_size, savedir, ext):
    # pylint: disable=too-many-arguments
    make_image_subdirs(model, savedir)

    sample_counter = collections.defaultdict(int)
    if isinstance(ext, str):
        ext = [ ext, ]

    for domain, data_it in enumerate(data_list):
        dump_single_domain_images(
            model, data_it, domain, n_eval, batch_size, savedir,
            sample_counter, ext
        )

# Custom Collate Function
def inference_collate_fn(batch):
    # batch: List[(Tensor, str)] → ([Tensor, Tensor, ...], [str, str, ...])
    images = [item[0] for item in batch]
    names  = [item[1] for item in batch]
    return images, names

def main():
    cmdargs = parse_cmdargs()

    args, model, evaldir = start_model_eval(
        cmdargs.model,
        cmdargs.epoch,
        cmdargs.model_state,
        merge_type = MERGE_NONE,
        batch_size = cmdargs.batch_size,
    )

    _override_dataset_paths(args.config.data, cmdargs.data_root)
    _coerce_domain_a_adjacent_z_pairs_to_cyclegan(args.config.data)
    inferred_domain = (
        None if not cmdargs.data_root else _infer_domain_from_split_dir(cmdargs.data_root)
    )
    if inferred_domain is not None:
        _restrict_to_domain(args.config.data, inferred_domain)
    else:
        _drop_missing_cyclegan_domains(args.config.data, cmdargs.split)

    data_list = construct_data_loaders(
        args.config.data, args.config.batch_size, split = cmdargs.split
    )

    # Set inference mode + patch DataLoader(s) with custom collate_fn
    if isinstance(data_list, (list, tuple)):
        new_list = []
        for dl in data_list:
            ds = dl.dataset
            if hasattr(ds, 'set_inference'):
                ds.set_inference(True)

            new_dl = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=dl.num_workers if hasattr(dl, "num_workers") else 0,
                collate_fn=inference_collate_fn
            )
            new_list.append(new_dl)
        data_list = new_list
    else:
        ds = data_list.dataset
        if hasattr(ds, 'set_inference'):
            ds.set_inference(True)

        data_list = [
            torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=data_list.num_workers if hasattr(data_list, "num_workers") else 0,
                collate_fn=inference_collate_fn
            )
        ]

    savedir = get_eval_savedir(
        evaldir, 'images', cmdargs.model_state, cmdargs.split
    )

    dump_images(
        model, data_list, cmdargs.n_eval, args.batch_size, savedir,
        cmdargs.ext
    )

if __name__ == '__main__':
    main() 
