"""浅层分类器：标准化 + SVM / 随机森林 / kNN / 逻辑回归。"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_classifier(name: str = "svm", seed: int = 42) -> Pipeline:
    """返回带标准化的 sklearn Pipeline。"""
    if name == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif name == "lr":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"unknown classifier: {name}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
