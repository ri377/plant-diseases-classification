"""Model architectures and Lightning modules."""

from plant_diseases_classification.models.cnn import PlantDiseaseCNN
from plant_diseases_classification.models.module import PlantDiseasesClassifier

__all__ = ["PlantDiseaseCNN", "PlantDiseasesClassifier"]
