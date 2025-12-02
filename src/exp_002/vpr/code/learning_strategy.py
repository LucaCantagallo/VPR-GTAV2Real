# learning_strategy.py

from triplet_loader import (
    get_triplet_dataloaders,
    refresh_triplet_dataloaders,
    run_triplet_epoch,
    get_triplet_loss,
)

from contrastive_loader import (
    get_contrastive_dataloaders,
    get_supcon_loss,
    run_supcon_epoch,
    refresh_contrastive_dataloaders,
)


def get_dataloaders(params):
    method = params["learning_method"]

    if method == "triplet":
        loaders = get_triplet_dataloaders(
            params["dataload"], params["train_dataset"], params["val_dataset"],
            params["train_samples_per_place"], params["valid_samples_per_place"],
            params, params["seed"]
        )
        return loaders

    if method == "infonce":
        loaders = get_contrastive_dataloaders(
            params["dataload"], params["train_dataset"], params["val_dataset"],
            params["train_samples_per_place"], params["valid_samples_per_place"],
            params, params["seed"]
        )
        return loaders

    raise NotImplementedError(f"Learning method {method} not implemented.")

def get_loss_fn(params):
    method = params["learning_method"]

    if method == "triplet":
        return get_triplet_loss()

    if method == "infonce":
        return get_supcon_loss(temperature=params["train"]["temperature"])

    raise NotImplementedError(f"Learning method {method} not implemented.")


def run_epoch(model, loader, loss_fn, optimizer, params, device):
    method = params["learning_method"]

    if method == "triplet":
        return run_triplet_epoch(model, loader, loss_fn, optimizer, True, device)

    if method == "infonce":
        return run_supcon_epoch(model, loader, loss_fn, optimizer, True, device)

    raise NotImplementedError(f"Learning method {method} not implemented.")


def run_epoch_valid(model, loader, loss_fn, params, device):
    method = params["learning_method"]

    if method == "triplet":
        return run_triplet_epoch(model, loader, loss_fn, None, False, device)

    if method == "infonce":
        return run_supcon_epoch(model, loader, loss_fn, None, False, device)

    raise NotImplementedError(f"Learning method {method} not implemented.")


def refresh_dataloaders(train_places, valid_places, params):
    method = params["learning_method"]

    if method == "triplet":
        return refresh_triplet_dataloaders(
            train_places, valid_places,
            params["train_samples_per_place"],
            params["valid_samples_per_place"],
            params
        )

    if method == "infonce":
        return refresh_contrastive_dataloaders(
            train_places, valid_places,
            params["train_samples_per_place"],
            params["valid_samples_per_place"],
            params
        )

    raise NotImplementedError(f"Learning method {method} not implemented.")
