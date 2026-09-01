import pandas as pd

from upgini.metadata import FeaturesMetadataV2
from upgini.utils.feature_info import FeatureInfo


def test_from_metadata_handles_duplicate_column_names():
    original = pd.DataFrame({"ip": ["1.1.1.1", "8.8.8.8", "9.9.9.9", "1.0.0.1"]})
    derived = pd.DataFrame({"ip": ["bin1", "bin2", "bin3", "bin4"]})
    data = pd.concat([original, derived], axis=1)
    assert isinstance(data["ip"], pd.DataFrame)

    feature_meta = FeaturesMetadataV2(
        name="ip",
        type="STRING",
        source="etalon",
        hit_rate=100.0,
        shap_value=0.0,
    )

    feature_info = FeatureInfo.from_metadata(feature_meta, data, is_client_feature=True, is_generated_feature=False)

    assert feature_info.value_preview
    assert "1.1.1.1" in feature_info.value_preview or "8.8.8.8" in feature_info.value_preview
