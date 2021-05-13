__all__ = ['sparse_synth_test_data_pipeline',
           'sparse_synth_test_data_poly_pipeline',
           'sparse_synth_test_data_noise_pipeline',
           'sparse_synth_test_data_rr_pipeline',
           ]

from kedro.pipeline import Pipeline, node

from .nodes import arrange_sparse_synth_test_data


def sparse_synth_test_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=arrange_sparse_synth_test_data,
                inputs="sparse_synth_test_data",
                outputs=["y", "X", "w", "y_true", "features_mask"],
                name="arrange_sparse_synth_test_data_node",
            ),
        ]
    )


def sparse_synth_test_data_poly_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=arrange_sparse_synth_test_data,
                inputs="sparse_synth_test_data_poly",
                outputs=["y", "X", "w", "y_true", "features_mask"],
                name="arrange_sparse_synth_test_data_node",
            ),
        ]
    )


def sparse_synth_test_data_noise_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=arrange_sparse_synth_test_data,
                inputs="sparse_synth_test_data_noise",
                outputs=["y", "X", "w", "y_true", "features_mask"],
                name="arrange_sparse_synth_test_data_node",
            ),
        ]
    )


def sparse_synth_test_data_rr_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=arrange_sparse_synth_test_data,
                inputs="sparse_synth_test_data_rr",
                outputs=["y", "X", "w", "y_true", "features_mask"],
                name="arrange_sparse_synth_test_data_node",
            ),
        ]
    )
