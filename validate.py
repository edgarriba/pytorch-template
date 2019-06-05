from tqdm import tqdm


def validate(epoch, model, criterion, data_loader, tb_writer, args):
    """Routine to validate an epoch.
    """
    model.eval()

    # dataset loop
    pbar = tqdm(enumerate(data_loader))
    loss = 0.0

    for batch_id, batch_data in pbar:
        # retrieve data from loader and copy data to device
        images = batch_data['images'].to(args.device)
        labels = batch_data['labels'].to(args.device)

        # inference model
        outputs = model(images)

	# compute the loss
        loss += criterion(outputs, labels)

    # print statistics at the end of the validation epoch
    if args.tensorboard:
        assert tb_writer is not None, "ERROR: tb_writer is None"
        global_step = len(data_loader) * epoch + batch_id
        tb_writer.add_scalar('val/loss', loss.detach(), global_step)


    # update logger bar
    pbar.set_description("## VALIDATION ## Epoch: {0} Batch: {1}/{2} Loss: {3:.4f}"
                         .format(epoch, batch_id, len(data_loader),
                                 loss.detach()))