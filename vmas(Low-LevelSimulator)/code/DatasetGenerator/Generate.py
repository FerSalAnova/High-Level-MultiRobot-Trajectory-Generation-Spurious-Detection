import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from DatasetGenerator.GeneratorBuilder import GeneratorBuilder

# gen = GeneratorBuilder("Training/configs/VMAS_navigation.yaml", {})
gen = GeneratorBuilder("DatasetGenerator/VMAS/config.yaml", {})
gen.generateTrainValTest()
del gen

