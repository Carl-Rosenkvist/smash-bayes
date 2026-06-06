from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass
class GaussianPCAEmulator:
    x_columns: list[str]
    y_columns: list[str]
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    pca: PCA
    gps: list[GaussianProcessRegressor]

    @classmethod
    def train(
        cls,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        n_components: int | float = 0.99,
    ) -> "GaussianPCAEmulator":
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = x_scaler.fit_transform(X)
        Y_scaled = y_scaler.fit_transform(Y)

        pca = PCA(n_components=n_components)
        Z = pca.fit_transform(Y_scaled)

        gps = []

        for i in range(Z.shape[1]):
            kernel = ConstantKernel(1.0) * RBF(
                length_scale=np.ones(X.shape[1])
            ) + WhiteKernel(noise_level=1e-6)

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=5,
            )

            gp.fit(X_scaled, Z[:, i])
            gps.append(gp)

        return cls(
            x_columns=list(X.columns),
            y_columns=list(Y.columns),
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            pca=pca,
            gps=gps,
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.x_columns]
        X_scaled = self.x_scaler.transform(X)

        Z_pred = np.column_stack([gp.predict(X_scaled) for gp in self.gps])

        Y_scaled_pred = self.pca.inverse_transform(Z_pred)
        Y_pred = self.y_scaler.inverse_transform(Y_scaled_pred)

        return pd.DataFrame(
            Y_pred,
            columns=self.y_columns,
            index=X.index,
        )

    def predict_row(self, parameters: dict[str, float]) -> pd.Series:
        X = pd.DataFrame([parameters])[self.x_columns]
        return self.predict(X).iloc[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "GaussianPCAEmulator":
        with Path(path).open("rb") as f:
            return pickle.load(f)
