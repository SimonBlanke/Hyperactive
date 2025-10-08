"""Experiment adapter for PyTorch Lightning experiments."""

from hyperactive.base import BaseExperiment
import torch
import lightning as L
from torch.utils.data import DataLoader


class TorchExperiment(BaseExperiment):
    """ Experiment adapter for PyTorch Lightning experiments.

    This class is used to perform experiments using PyTorch Lightning modules.
    It allows for hyperparameter tuning and evaluation of the model's performance
    using specified metrics.

    The experiment trains a Lightning module with given hyperparameters and returns
    the validation metric value for optimization.

    Parameters
    ----------
    datamodule : L.LightningDataModule
        A PyTorch Lightning DataModule that handles data loading and preparation.
    lightning_module : type
        A PyTorch Lightning Module class (not an instance) that will be instantiated
        with hyperparameters during optimization.
    trainer : L.Trainer
        A PyTorch Lightning Trainer that handles model training and evaluation.
    objective_metric : str, optional (default='val_loss')
        The metric used to evaluate the model's performance. This should correspond
        to a metric logged in the LightningModule during validation.

    Examples
    --------
    >>> from hyperactive.experiment.integrations import TorchExperiment
    >>> import torch
    >>> import lightning as L
    >>> from torch import nn
    >>> from torch.utils.data import DataLoader
    >>> 
    >>> # Define a simple Lightning Module
    >>> class SimpleLightningModule(L.LightningModule):
    ...     def __init__(self, input_dim=10, hidden_dim=16, lr=1e-3):
    ...         super().__init__()
    ...         self.save_hyperparameters()
    ...         self.model = nn.Sequential(
    ...             nn.Linear(input_dim, hidden_dim),
    ...             nn.ReLU(),
    ...             nn.Linear(hidden_dim, 2)
    ...         )
    ...         self.lr = lr
    ...     
    ...     def forward(self, x):
    ...         return self.model(x)
    ...     
    ...     def training_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         loss = nn.functional.cross_entropy(y_hat, y)
    ...         self.log("train_loss", loss)
    ...         return loss
    ...     
    ...     def validation_step(self, batch, batch_idx):
    ...         x, y = batch
    ...         y_hat = self(x)
    ...         val_loss = nn.functional.cross_entropy(y_hat, y)
    ...         self.log("val_loss", val_loss, on_epoch=True)
    ...         return val_loss
    ...     
    ...     def configure_optimizers(self):
    ...         return torch.optim.Adam(self.parameters(), lr=self.lr)
    >>> 
    >>> # Create DataModule
    >>> class RandomDataModule(L.LightningDataModule):
    ...     def __init__(self, batch_size=32):
    ...         super().__init__()
    ...         self.batch_size = batch_size
    ...     
    ...     def setup(self, stage=None):
    ...         dataset = torch.utils.data.TensorDataset(
    ...             torch.randn(100, 10), 
    ...             torch.randint(0, 2, (100,))
    ...         )
    ...         self.train, self.val = torch.utils.data.random_split(
    ...             dataset, [80, 20]
    ...         )
    ...     
    ...     def train_dataloader(self):
    ...         return DataLoader(self.train, batch_size=self.batch_size)
    ...     
    ...     def val_dataloader(self):
    ...         return DataLoader(self.val, batch_size=self.batch_size)
    >>> 
    >>> # Setup experiment
    >>> trainer = L.Trainer(max_epochs=3, enable_progress_bar=False)
    >>> datamodule = RandomDataModule()
    >>> 
    >>> experiment = TorchExperiment(
    ...     datamodule=datamodule,
    ...     lightning_module=SimpleLightningModule,
    ...     trainer=trainer,
    ...     objective_metric="val_loss"
    ... )
    >>> 
    >>> params = {"input_dim": 10, "hidden_dim": 16, "lr": 1e-3}
    >>> 
    >>> val_result, metadata = experiment._evaluate(params)
    """

    def __init__(self, datamodule, lightning_module, trainer, objective_metric: str = "val_loss"):
        # todo: write any hyper-parameters to self
        self.datamodule = datamodule
        self.lightning_module = lightning_module
        self.trainer = trainer
        self.objective_metric = objective_metric
  
        super().__init__()

    def _paramnames(self):
        """Return the parameter names of the search.

        Returns
        -------
        list of str, or None
            The parameter names of the search parameters.
            If not known or arbitrary, return None.
        """
        import inspect
        sig = inspect.signature(self.lightning_module.__init__)
        return [p for p in sig.parameters.keys() if p != 'self']

    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """

        model = self.lightning_module(**params)
        self.trainer.fit(model, self.datamodule)
        # get validation results
        if self.objective_metric not in self.trainer.callback_metrics:
            raise ValueError(f"objective metric {self.objective_metric} not found in trainer callback metrics")
        else:
            val_results = self.trainer.callback_metrics.get(self.objective_metric, float('inf'))
        metadata = {}
        import numpy as np
        return np.float64(val_results.detach().cpu().item()), metadata

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        from torch import nn
        import lightning as L
        from lightning import Trainer
        
        class SimpleLightningModule(L.LightningModule):
            def __init__(self, input_dim=10, hidden_dim=16, lr=1e-3):
                super().__init__()
                self.save_hyperparameters()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 2)
                )
                self.lr = lr

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = nn.functional.cross_entropy(y_hat, y)
                self.log("train_loss", loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                val_loss = nn.functional.cross_entropy(y_hat, y)
                self.log("val_loss", val_loss, on_epoch=True)
                return val_loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=self.lr)

        class RandomDataModule(L.LightningDataModule):
            def __init__(self, batch_size=32):
                super().__init__()
                self.batch_size = batch_size

            def setup(self, stage=None):
                dataset = torch.utils.data.TensorDataset(
                    torch.randn(100, 10), 
                    torch.randint(0, 2, (100,))
                )
                self.train, self.val = torch.utils.data.random_split(
                    dataset, [80, 20]
                )

            def train_dataloader(self):
                return DataLoader(self.train, batch_size=self.batch_size)

            def val_dataloader(self):
                return DataLoader(self.val, batch_size=self.batch_size)

        trainer = Trainer(
            max_epochs=1, 
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False
        )
        datamodule = RandomDataModule(batch_size=16)

        params = {
            "datamodule": datamodule,
            "lightning_module": SimpleLightningModule,
            "trainer": trainer,
            "objective_metric": "val_loss"
        }
        
        return params

    @classmethod
    def _get_score_params(self):
        """Return settings for testing score/evaluate functions.

        Returns a list, the i-th element should be valid arguments for
        self.evaluate and self.score, of an instance constructed with
        self.get_test_params()[i].

        Returns
        -------
        list of dict
            The parameters to be used for scoring.
        """
        score_params1 = {"input_dim": 10, "hidden_dim": 20, "lr": 0.001}
        score_params2 = {"input_dim": 10, "hidden_dim": 16, "lr": 0.01}
        return [score_params1, score_params2]