import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from upgini.metadata import FeaturesMetadataV2
from upgini.resource_bundle import ResourceBundle

LLM_SOURCE = "LLM with external data augmentation"


@dataclass
class FeatureInfo:
    name: str
    internal_name: str
    rounded_shap: float
    hitrate: float
    value_preview: str
    provider: str
    internal_provider: str
    source: str
    internal_source: str
    update_frequency: str
    commercial_schema: str
    doc_link: str
    data_provider_link: str
    data_source_link: str
    psi_value: Optional[float] = None

    @staticmethod
    def from_metadata(
        feature_meta: FeaturesMetadataV2, data: Optional[pd.DataFrame], is_client_feature: bool
    ) -> "FeatureInfo":
        return FeatureInfo(
            name=_get_name(feature_meta),
            internal_name=_get_internal_name(feature_meta),
            rounded_shap=_round_shap_value(feature_meta.shap_value),
            hitrate=feature_meta.hit_rate,
            value_preview=_get_feature_sample(feature_meta, data),
            provider=_get_provider(feature_meta, is_client_feature),
            internal_provider=_get_internal_provider(feature_meta, is_client_feature),
            source=_get_source(feature_meta, is_client_feature),
            internal_source=_get_internal_source(feature_meta, is_client_feature),
            update_frequency=feature_meta.update_frequency,
            commercial_schema=feature_meta.commercial_schema,
            doc_link=feature_meta.doc_link,
            data_provider_link=feature_meta.data_provider_link,
            data_source_link=feature_meta.data_source_link,
            psi_value=feature_meta.psi_value,
        )

    def to_row(self, bundle: ResourceBundle) -> Dict[str, str]:
        return {
            bundle.get("features_info_name"): self.name,
            bundle.get("features_info_shap"): self.rounded_shap,
            bundle.get("features_info_psi"): self.psi_value,
            bundle.get("features_info_hitrate"): self.hitrate,
            bundle.get("features_info_value_preview"): self.value_preview,
            bundle.get("features_info_provider"): self.provider,
            bundle.get("features_info_source"): self.source,
            bundle.get("features_info_update_frequency"): self.update_frequency,
        }

    def to_row_without_links(self, bundle: ResourceBundle) -> Dict[str, str]:
        return {
            bundle.get("features_info_name"): self.internal_name,
            bundle.get("features_info_shap"): self.rounded_shap,
            bundle.get("features_info_psi"): self.psi_value,
            bundle.get("features_info_hitrate"): self.hitrate,
            bundle.get("features_info_value_preview"): self.value_preview,
            bundle.get("features_info_provider"): self.internal_provider,
            bundle.get("features_info_source"): self.internal_source,
            bundle.get("features_info_update_frequency"): self.update_frequency,
        }

    def to_internal_row(self, bundle: ResourceBundle) -> Dict[str, str]:
        return {
            bundle.get("features_info_name"): self.internal_name,
            "feature_link": self.doc_link,
            bundle.get("features_info_shap"): self.rounded_shap,
            bundle.get("features_info_psi"): self.psi_value,
            bundle.get("features_info_hitrate"): self.hitrate,
            bundle.get("features_info_value_preview"): self.value_preview,
            bundle.get("features_info_provider"): self.internal_provider,
            "provider_link": self.data_provider_link,
            bundle.get("features_info_source"): self.internal_source,
            "source_link": self.data_source_link,
            bundle.get("features_info_commercial_schema"): self.commercial_schema or "",
            bundle.get("features_info_update_frequency"): self.update_frequency,
        }


def _get_feature_sample(feature_meta: FeaturesMetadataV2, data: Optional[pd.DataFrame]) -> str:
    if data is not None and len(data) > 0 and feature_meta.name in data.columns:
        if len(data) > 3:
            rand = np.random.RandomState(42)
            feature_sample = rand.choice(data[feature_meta.name].dropna().unique(), 3).tolist()
        else:
            feature_sample = data[feature_meta.name].dropna().unique().tolist()
        if len(feature_sample) > 0 and isinstance(feature_sample[0], float):
            feature_sample = [round(f, 4) for f in feature_sample]
        feature_sample = [str(f) for f in feature_sample]
        feature_sample = ", ".join(feature_sample)
        if len(feature_sample) > 30:
            feature_sample = feature_sample[:30] + "..."
    else:
        feature_sample = ""
    return feature_sample


def _get_name(feature_meta: FeaturesMetadataV2) -> str:
    if feature_meta.doc_link:
        return _to_anchor(feature_meta.doc_link, feature_meta.name)
    else:
        return feature_meta.name


def _get_internal_name(feature_meta: FeaturesMetadataV2) -> str:
    return feature_meta.name


def _get_provider(feature_meta: FeaturesMetadataV2, is_client_feature: bool) -> str:
    providers = _list_or_single(feature_meta.data_providers, feature_meta.data_provider)
    provider_links = _list_or_single(feature_meta.data_provider_links, feature_meta.data_provider_link)
    if providers:
        provider = _make_links(providers, provider_links)
    else:
        provider = "" if is_client_feature else _to_anchor("https://upgini.com", "Upgini")
    return provider


def _get_internal_provider(feature_meta: FeaturesMetadataV2, is_client_feature: bool) -> str:
    providers = _list_or_single(feature_meta.data_providers, feature_meta.data_provider)
    if providers:
        return ", ".join(providers)
    else:
        return "" if is_client_feature else (feature_meta.data_provider or "Upgini")


def _get_source(feature_meta: FeaturesMetadataV2, is_client_feature: bool) -> str:
    sources = _list_or_single(feature_meta.data_sources, feature_meta.data_source)
    source_links = _list_or_single(feature_meta.data_source_links, feature_meta.data_source_link)
    if sources:
        source = _make_links(sources, source_links)
    else:
        source = _get_internal_source(feature_meta, is_client_feature)
    return source


def _get_internal_source(feature_meta: FeaturesMetadataV2, is_client_feature: bool) -> str:
    sources = _list_or_single(feature_meta.data_sources, feature_meta.data_source)
    if sources:
        return ", ".join(sources)
    else:
        return feature_meta.data_source or (
            LLM_SOURCE
            if not feature_meta.name.endswith("_country")
            and not feature_meta.name.endswith("_postal_code")
            and not is_client_feature
            else ""
        )


def _list_or_single(lst: List[str], single: str):
    return lst or ([single] if single else [])


def _to_anchor(link: str, value: str) -> str:
    if not value:
        return ""
    elif not link:
        return value
    elif value == LLM_SOURCE:
        return value
    else:
        return f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{value}</a>"


def _make_links(names: List[str], links: List[str]) -> str:
    all_links = [_to_anchor(link, name) for name, link in itertools.zip_longest(names, links)]
    return ",".join(all_links)


def _round_shap_value(shap: float) -> float:
    if shap >= 0.0 and shap < 0.0001:
        return 0.0001
    else:
        return round(shap, 4)
