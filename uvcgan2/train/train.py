from itertools import islice
import tqdm

from uvcgan2.config      import Args
from uvcgan2.data        import construct_data_loaders
from uvcgan2.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan2.cgan        import construct_model
from uvcgan2.utils.log   import setup_logging

from .metrics   import LossMetrics
from .callbacks import TrainingHistory
from .transfer  import transfer

from uvcgan2.data.adjacent_pair_dataset import AdjacentZPairDataset

import os
from torchvision.utils import save_image

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wandb = None

def training_epoch(it_train, model, title, steps_per_epoch):
    model.train()

    steps = len(it_train)
    if steps_per_epoch is not None:
        steps = min(steps, steps_per_epoch)

    progbar = tqdm.tqdm(desc = title, total = steps, dynamic_ncols = True)
    metrics = LossMetrics()

    for batch in islice(it_train, steps):
        model.set_input(batch)
        model.optimization_step()

        metrics.update(model.get_current_losses())

        progbar.set_postfix(metrics.values, refresh = False)
        progbar.update()

    progbar.close()
    return metrics

def try_continue_training(args, model):
    history = TrainingHistory(args.savedir)

    start_epoch = model.find_last_checkpoint_epoch()
    model.load(start_epoch)

    if start_epoch > 0:
        history.load()

    start_epoch = max(start_epoch, 0)

    return (start_epoch, history)

def train(args_dict):
    args = Args.from_args_dict(**args_dict)

    setup_logging(args.log_level)
    seed_everything(args.config.seed)

    device   = get_torch_device_smart()
    it_train = construct_data_loaders(
        args.config.data, args.config.batch_size, split = 'train'
    )

    print("Starting training...")
    print(args.config.to_json(indent = 4))

    model = construct_model(
        args.savedir, args.config, is_train = True, device = device
    )
    start_epoch, history = try_continue_training(args, model)

    if (start_epoch == 0) and (args.transfer is not None):
        transfer(model, args.transfer)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        title   = 'Epoch %d / %d' % (epoch, args.epochs)
        metrics = training_epoch(
            it_train, model, title, args.config.steps_per_epoch
        )

        history.end_epoch(epoch, metrics)
        model.end_epoch(epoch)

        if wandb is not None and getattr(wandb, "run", None) is not None:
            log_dict = dict(metrics.values)
            log_dict['epoch'] = epoch
            # Helpful to track actual optimizer LR if schedulers are used.
            try:
                log_dict['lr_gen'] = model.optimizers.gen.param_groups[0]['lr']
                log_dict['lr_disc'] = model.optimizers.disc.param_groups[0]['lr']
            except Exception:
                pass
            wandb.log(log_dict, step=epoch)

        if epoch % args.checkpoint == 0:
            model.save(epoch)

    model.save(epoch = None)

# Save images of pairs to debug. 
def save_debug_image_pairs(batch, save_dir='debug_pairs', max_pairs=4):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(max_pairs, len(batch['A']))):
        a_img = batch['A'][i]
        b_img = batch['B'][i]
        a_name = batch['A_name'][i]
        b_name = batch['B_name'][i]

        # Remove file extension from names and replace '=' with '-' for compatibility
        a_name_clean = os.path.splitext(a_name)[0].replace('=', '-')
        b_name_clean = os.path.splitext(b_name)[0].replace('=', '-')

        print(save_dir)

        save_image(a_img, os.path.join(save_dir, f'{i}_A_{a_name_clean}.png'))
        save_image(b_img, os.path.join(save_dir, f'{i}_B_{b_name_clean}.png'))
