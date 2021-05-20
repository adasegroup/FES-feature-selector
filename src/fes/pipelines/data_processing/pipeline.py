from kedro.pipeline import Pipeline, node

from .nodes import arrange_synth_test_data


# Here now is only one pipeline for synthetic dataset creation, configured during the run or in the parameter file
def synth_test_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=arrange_synth_test_data,
                inputs="parameters",
                outputs=["y", "X", "w", "y_true", "features_mask"],
                name="synth_test_data_node",
            ),
        ]
    )
