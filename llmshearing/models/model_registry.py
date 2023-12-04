# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from llmshearing.models.composer_llama import ComposerMosaicLlama
from llmshearing.models.composer_pythia import ComposerMosaicPythia

COMPOSER_MODEL_REGISTRY = {
    'mosaic_llama_125m': ComposerMosaicLlama,
    'mosaic_llama_370m': ComposerMosaicLlama,
    "mosaic_llama_1.3b": ComposerMosaicLlama,
    "mosaic_llama_3b": ComposerMosaicLlama,
    'mosaic_llama_7b': ComposerMosaicLlama,
    'mosaic_llama_13b': ComposerMosaicLlama,
    'mosaic_llama_30b': ComposerMosaicLlama,
    'mosaic_llama_65b': ComposerMosaicLlama,
    'mosaic_pythia_70m': ComposerMosaicPythia,
    'mosaic_pythia_160m': ComposerMosaicPythia,
    'mosaic_pythia_410m': ComposerMosaicPythia,
    'mosaic_pythia_1.4b': ComposerMosaicPythia,
    'mosaic_llama2_370m': ComposerMosaicLlama,
    "mosaic_llama2_1.3b": ComposerMosaicLlama,
    "mosaic_llama2_3b": ComposerMosaicLlama,
    'mosaic_llama2_7b': ComposerMosaicLlama,
    'mosaic_llama2_13b': ComposerMosaicLlama,
    'mosaic_together_3b': ComposerMosaicPythia 
}
