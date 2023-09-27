from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.utils import save_checkpoint
import torch.distributed as dist

def move_to_device(batch):
    for key in batch:
        if type(batch[key]) is torch.Tensor:
            batch[key] = batch[key].cuda()
    return batch

def train(
        args,
        rank,
        model,
        dataloders,
        optims,
        schedulers,
        logger,
        ):
    # total_batch_size = args.total_batch_size
    train_dataloader, vail_dataloader = dataloders
    if rank == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.data.train_batch_size}")
        # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.train.max_train_steps}")

    global_step = 0
    first_epoch = 0
    # progress_bar = tqdm(range(global_step, args.max_train_steps), disable = not accelerator.is_local_main_process)
    progress_bar = tqdm(range(global_step, args.train.max_train_steps))
    progress_bar.set_description("Steps")

    
    for epoch in range(first_epoch, args.train.num_train_epochs):
        model.train()
        # dataloders.sampler.set_epoch(epoch)
        train_dataloader.sampler.set_epoch(epoch)
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            # batch = batch.to(args.train.device)
            if rank == 0:
                global_step += 1
            batch = move_to_device(batch=batch)
            if step % args.train.gradient_accumulation_steps == 0:
                progress_bar.update(1)

            model_pred, noise , latents, timesteps = model(batch)

            if model.module.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif model.module.noise_scheduler.config.prediction_type == "v_prediction":
                target = model.module.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {model.noise_scheduler.config.prediction_type}")
            
            if args.train.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                )
                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                # rebalance the sample-wise losses with their respective loss weights.
                # Finally, we take the mean of the rebalanced loss.
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

            
            loss.backward()

            reduced_loss = loss.data.clone()
            dist.all_reduce(reduced_loss)
            reduced_loss = reduced_loss / torch.cuda.device_count()

            optims.step()
            schedulers.step()
            optims.zero_grad()

            logs = {"step_loss": reduced_loss.detach().item(), "lr": schedulers.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


            if global_step % args.train.checkpointing_steps == 0:
                if rank == 0:
                    save_checkpoint(model=model,args=args,global_step=global_step)

