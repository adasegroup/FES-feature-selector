from kedro.pipeline import Pipeline, node

from .nodes import fit_model, evaluate_perm_importance
from .nodes import evaluate_iht


def perm_importance_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=fit_model,
                inputs=["y", "X"],
                outputs="regressor",
                name="fit_model_node",
            ),
            node(
                func=evaluate_perm_importance,
                inputs=["regressor", "y", "X", "w", "y_true", "features_mask", "parameters"],
                outputs=None,
                name="evaluate_perm_importance_node",
            ),
        ]
    )


def iht_pipeline(**kwargs):
    return Pipeline([
        node(
            func=evaluate_iht,
            inputs=["y", "X", "w", "y_true", "features_mask", "parameters"],
            outputs=None,
            name="evaluate_iht_node"
        )
    ])
