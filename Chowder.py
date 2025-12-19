# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################


import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

import warnings
from sklearn.metrics import roc_auc_score, f1_score
from Code_py import BalancedBatchSampler


def slide_level_train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training step for slide-level experiments. This will serve as the
    ``train_step`` in ``TorchTrainer``printclass.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    train_dataloader: torch.utils.data.DataLoader
        Training data loader.
    criterion: nn.Module
        The loss criterion used for training.
    optimizer: Callable = Adam
        The optimizer class to use.
    device : str = "cpu"
        The device to use for training and evaluation.
    """
    model.train()

    _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

    for batch in train_dataloader:
        # Get data.
        features, labels = batch
        # Put on device.
        features = features.to(device)
        labels = labels.to(torch.int64).to(device)
        # Compute logits and loss.
        logits = model(features)
        loss = criterion(logits[:,0], labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Stack logits & labels to compute epoch metrics.
        _epoch_loss.append(loss.detach().cpu().numpy())
        _epoch_logits.append(logits.detach())
        _epoch_labels.append(labels.detach())

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels


def slide_level_val_step(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference step for slide-level experiments. This will serve as the
    ``val_step`` in ``TorchTrainer``class.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    val_dataloader: torch.utils.data.DataLoader
        Inference data loader.
    criterion: nn.Module
        The loss criterion used for training.
    device : str = "cpu"
        The device to use for training and evaluation.
    """
    model.eval()

    with torch.no_grad():
        _epoch_loss, _epoch_logits, _epoch_labels = [], [], []

        for batch in val_dataloader:
            # Get data.
            features, labels = batch

            # Put on device.
            features = features.to(device)
            labels = labels.to(torch.int64).to(device)

            # Compute logits and loss.
            logits = model(features)
            loss = criterion(logits[:,0], labels)

            # Stack logits & labels to compute epoch metrics.
            _epoch_loss.append(loss.detach().cpu().numpy())
            _epoch_logits.append(logits.detach())
            _epoch_labels.append(labels.detach())

    _epoch_loss = np.mean(_epoch_loss)
    _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_labels

class TorchTrainer:
    """Trainer class for training and evaluating PyTorch models.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    criterion: nn.Module
        The loss criterion used for training.
    metrics: Dict[str, Callable]
        Dictionary of metrics functions to evaluate the model's performance.
    batch_size: int = 16
        The batch size for training and evaluation
    num_epochs : int = 10
        The number of training epochs.
    learning_rate: float = 1.0e-3
        The learning rate for the optimizer.
    weight_decay: float = 0.0
        The weight decay for the optimizer.
    device : str = "cpu"
        The device to use for training and evaluation.
    num_workers: int = 8
        Number of workers.
    balanced: bool = False
        If you want your batchs to be balanced in the number of each class
    optimizer: Callable = Adam
        The optimizer class to use.
    train_step: Callable = slide_level_train_step
        The function for training step.
    val_step: Callable = slide_level_val_step
        The function for validation step.

    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metrics: Dict[str, Callable],
        batch_size: int = 16,
        num_epochs: int = 10,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        num_workers: int = 8,
        balanced : bool = False,
        optimizer: Callable = Adam,
        train_step: Callable = slide_level_train_step,
        val_step: Callable = slide_level_val_step,
        nb_classes:int = 2,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.balanced = balanced

        self.train_step = train_step
        self.val_step = val_step

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.device = device
        self.num_workers = num_workers

        self.nb_classes=nb_classes
        self.train_losses: List[float]
        self.val_losses: List[float]
        self.train_metrics: Dict[str, List[float]]
        self.val_metrics: Dict[str, List[float]]

    def train(
        self,
        train_set: Subset,
        test_set: Subset,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Train the model using the provided training and validation datasets.

        Parameters
        ----------
        train_set: Subset
            The training dataset.
        test_set: Subset
            The validation dataset.

        Returns
        -------
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]
            2 dictionaries containing the training and validation metrics for each epoch.
        """
        # Dataloaders.
        if self.balanced:
            sampler_train= BalancedBatchSampler(train_set, bs=self.batch_size, labels=[int(train_set[i][1]) for i in range(len(train_set))])
            train_dataloader = DataLoader(
                dataset=train_set,
                shuffle=False,
                batch_size=self.batch_size,
                pin_memory=False,
                sampler=sampler_train,
                drop_last=True,
                num_workers=self.num_workers,
            )
            sampler_val=BalancedBatchSampler(test_set,  bs=self.batch_size, labels=[int(train_set[i][1]) for i in range(len(test_set))])
            val_dataloader = DataLoader(
                dataset=test_set,
                shuffle=False,
                batch_size=self.batch_size,
                pin_memory=False,
                sampler=sampler_val,
                drop_last=False,
                num_workers=self.num_workers,
            )
        else:
            train_dataloader = DataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=False,
                drop_last=True,
                num_workers=self.num_workers,
            )
            val_dataloader = DataLoader(
                dataset=test_set,
                shuffle=True,
                batch_size=self.batch_size,
                pin_memory=False,

                drop_last=False,
                num_workers=self.num_workers,
            )

        # Prepare modules.
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)
        optimizer = self.optimizer(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Training.
        train_losses, val_losses = [], []
        train_metrics: Dict[str, List[float]] = {
            k: [] for k in self.metrics.keys()
        }
        val_metrics: Dict[str, List[float]] = {
            k: [] for k in self.metrics.keys()
        }
        for ep in range(self.num_epochs):
            # Train step.
            (
                train_epoch_loss,
                train_epoch_logits,
                train_epoch_labels,
            ) = self.train_step(
                model=model,
                train_dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
            )

            # Inference step.
            val_epoch_loss, val_epoch_logits, val_epoch_labels = self.val_step(
                model=model,
                val_dataloader=val_dataloader,
                criterion=criterion,
                device=self.device,
            )
            
            # Compute metrics.
            for k, m in self.metrics.items():
                train_metric = m(train_epoch_labels, train_epoch_logits,self.nb_classes)
                val_metric = m(val_epoch_labels, val_epoch_logits,self.nb_classes)

                train_metrics[k].append(train_metric)
                val_metrics[k].append(val_metric)

                print(
                    f"Epoch {ep+1}: train_loss={train_epoch_loss:.5f}, train_{k}={train_metric:.4f}, val_loss={val_epoch_loss:.5f}, val_{k}={val_metric:.4f}"
                )

            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        return train_metrics, val_metrics, train_losses, val_losses

    def evaluate(
        self,
        test_set: Subset,
    ) -> Dict[str, float]:
        """Evaluate the model using the provided test dataset.

        Parameters
        ----------
        test_set: Subset
            The test dataset.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the test metrics.
        """
        # Dataloader.
        test_dataloader = DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # Prepare modules.
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)

        # Inference step.
        _, test_epoch_logits, test_epoch_labels = self.val_step(
            model=model,
            val_dataloader=test_dataloader,
            criterion=criterion,
            device=self.device,
        )

        # Compute metrics.
        test_metrics = {
            k: m(test_epoch_labels, test_epoch_logits, self.nb_classes)
            for k, m in self.metrics.items()
        }

        return test_metrics

    def predict(
        self,
        test_set: Subset,
    ) -> Tuple[np.array, np.array]:
        """Make predictions using the provided test dataset.

        Parameters
        ----------
        test_set: Subset
            The test dataset.

        Returns
        --------
        Tuple[np.array, np.array]
            A tuple containing the test labels and logits.
        """
        # Dataloader
        test_dataloader = DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        # Prepare modules
        model = self.model.to(self.device)
        criterion = self.criterion.to(self.device)

        # Val step
        _, test_epoch_logits, test_epoch_labels = self.val_step(
            model=model,
            val_dataloader=test_dataloader,
            criterion=criterion,
            device=self.device,
        )

        return test_epoch_labels, test_epoch_logits

class MLP(torch.nn.Sequential):
    """MLP Module.

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Dimension of hidden layer(s).
    dropout: Optional[List[float]] = None
        Dropout rate(s).
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        MLP activation.
    bias: bool = True
        Add bias to MLP hidden layers.

    Raises
    ------
    ValueError
        If ``hidden`` and ``dropout`` do not share the same length.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Softmax(),
        bias: bool = True,
    ):
        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)

class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):  # pylint: disable=arguments-renamed
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, SEQ_LEN, IN_FEATURES).
        mask: Optional[torch.BoolTensor] = None
            True for values that were padded, shape (B, SEQ_LEN, 1),

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )

class TilesMLP(torch.nn.Module):
    """MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    dropout: Optional[torch.nn.Module] = None
        Optional dropout module. Will be interlaced with the linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout: Optional[torch.nn.Module] = None,
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = h

        self.hidden_layers.append(
            torch.nn.Linear(in_features, out_features, bias=bias)
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x

class ExtremeLayer(torch.nn.Module):
    """Extreme layer.
    Returns concatenation of n_top top tiles and n_bottom bottom tiles
    .. warning::
        If top tiles or bottom tiles is superior to the true number of
        tiles in the input then padded tiles will be selected and their value
        will be 0.
    Parameters
    ----------
    n_top: Optional[int] = None
        Number of top tiles to select
    n_bottom: Optional[int] = None
        Number of bottom tiles to select
    dim: int = 1
        Dimension to select top/bottom tiles from
    return_indices: bool = False
        Whether to return the indices of the extreme tiles

    Raises
    ------
    ValueError
        If ``n_top`` and ``n_bottom`` are set to ``None`` or both are 0.
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not (
            (n_top is not None and n_top > 0)
            or (n_bottom is not None and n_bottom > 0)
        ):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, N_TILES, IN_FEATURES).
        mask: Optional[torch.BoolTensor]
            True for values that were padded, shape (B, N_TILES, 1).

        Warnings
        --------
        If top tiles or bottom tiles is superior to the true number of tiles in
        the input then padded tiles will be selected and their value will be 0.

        Returns
        -------
        values: torch.Tensor
            Extreme tiles, shape (B, N_TOP + N_BOTTOM).
        indices: torch.Tensor
            If ``self.return_indices=True``, return extreme tiles' indices.
        """

        if (
            self.n_top
            and self.n_bottom
            and ((self.n_top + self.n_bottom) > x.shape[self.dim])
        ):
            warnings.warn(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. "
                f"Values will appear twice (in top and in bottom)"
            )
        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(
                    k=self.n_top, sorted=True, dim=self.dim
                )
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    warnings.warn(
                        "The top tiles contain masked values, they will be set to zero."
                    )
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    warnings.warn(
                        "The bottom tiles contain masked values, they will be set to zero."
                    )
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

    def extra_repr(self) -> str:
        """Format representation."""
        return f"n_top={self.n_top}, n_bottom={self.n_bottom}"
        

class Chowder(nn.Module):
    """Chowder MIL model (See [1]_).

    Example:
        >>> module = Chowder(in_features=128, out_features=1, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int
        Controls the number of scores and, by extension, the number of out_features.
    n_top: int
        Number of tiles with hightest scores that are selected and fed to the MLP.
    n_bottom: int
        Number of tiles with lowest scores that are selected and fed to the MLP.
    tiles_mlp_hidden: Optional[List[int]] = None
        Number of units for layers in the first MLP applied tile wise to compute
        a score for each tiles from the tile features.
        If `None`, a linear layer is used to compute tile scores.
        If e.g. `[128, 64]`, the tile scores are computed with a MLP of dimension
        features_dim -> 128 -> 64 -> 1.
    mlp_hidden: Optional[List[int]] = None
        Number of units for layers of the second MLP that combine top and bottom
        scores and outputs a final prediction at the slide-level. If `None`, a
        linear layer is used to compute the prediction from the extreme scores.
        If e.g. `[128, 64]`, the prediction is computed
        with a MLP n_top + n_bottom -> 128 -> 64 -> 1.
    mlp_dropout: Optional[List[float]] = None
        Dropout that is used for each layer of the MLP. If `None`, no dropout
        is used.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation that is used after each layer of the MLP.
    bias: bool = True
        Whether to add bias for layers of the tiles MLP.

    References
    ----------
    .. [1] Pierre Courtiol, Eric W. Tramel, Marc Sanselme, and Gilles Wainrib. Classification
    and disease localization in histopathology using only global labels: A weakly-supervised
    approach. CoRR, abs/1802.02212, 2018.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        return_indices: bool =False,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ) -> None:
        super(Chowder, self).__init__()
        if n_top is None and n_bottom is None:
            raise ValueError(
                "At least one of `n_top` or `n_bottom` must not be None."
            )

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=1,
        )
        
        self.score_model.apply(self.weight_initialization)
        self.return_indices=return_indices
        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom, return_indices=self.return_indices)
        
        
        mlp_in_features = n_top + n_bottom
        self.mlp = MLP(
            mlp_in_features,
            out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        
        self.mlp.apply(self.weight_initialization)

    @staticmethod
    def weight_initialization(module: torch.nn.Module) -> None:
        """Initialize weights for the module using Xavier initialization method,
        "Understanding the difficulty of training deep feedforward neural networks",
        Glorot, X. & Bengio, Y. (2010)."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=features[..., :], mask=mask)
        if self.return_indices:
            extreme_scores,extreme_indices = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        else:
            extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        # Apply MLP to the N_TOP + N_BOTTOM scores.
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)
        
        if self.return_indices:
            return (y.squeeze(2), extreme_indices)
        else:
            return y.squeeze(2)


def auc2(labels: np.array, logits: np.array, nb_classes:int) -> float:
    """ROC AUC score for binary classification.
    Parameters
    ----------
    labels: np.array
        Labels of the outcome.
    logits: np.array
        Probabilities.
    """
    essai=torch.nn.functional.softmax(torch.tensor(logits)[:,0])
    labels= torch.nn.functional.one_hot(torch.tensor(labels).to(torch.int64), num_classes=nb_classes)
    return roc_auc_score(labels, essai, average='micro',multi_class='ovr')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_cv_metrics(
    cv_metrics: List[Dict[str, float]], epoch: int = -1
    ) -> Dict[str, float]:
    """Get mean and std from cross-validation metrics at a given epoch."""
    cv_mean_metrics = {}
    metrics_names = cv_metrics[0].keys()
    for m_name in metrics_names:
        values = [fold_metrics[m_name][epoch] for fold_metrics in cv_metrics]
        mean_metric, std_metric = np.mean(values), np.std(values)
        cv_mean_metrics[m_name] = f"{mean_metric:.4f} Â± {std_metric:.4f}"
    return cv_mean_metrics