# Inferring Sensitive Attributed from Model Explanations

Code for the paper titled "Inferring Sensitive Attributes from Model Explanations" published in ACM CIKM 2022.

## Requirements

You need __conda__. Create a virtual environment and install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate attinf-explanations
```

To update the env:

```bash
conda env update --name attinf-explanations --file environment.yml
```

or

```bash
conda activate attinf-explanations
conda env update --file environment.yml
```

## Dataset

Link to datasets: https://drive.google.com/drive/folders/1bUH02Y9I6_NVrfo5_8PwWtdklk15rXPJ

## Usage


### Evaluate attribute inference attacks of against explanations

```bash
python -m src.attribute_inference --dataset {LAW,MEPS,CENSUS,CREDIT,COMPAS} --explanations {IntegratedGradients,smoothgrad,DeepLift,GradientShap} --attfeature {both,expl}
```
attfeature evaluates the attacks on only explanations (expl) or both predictions and explanations (both)

### Attacking using entire explanations for both sensitive and non-sensitive attributes

```bash
python -m src.attribute_inference --dataset {LAW,MEPS,CENSUS,CREDIT,COMPAS} --explanations {IntegratedGradients,smoothgrad,DeepLift,GradientShap} --attfeature expl --with_sattr True
```

### Attacking using only explanations corresponding to sensitive attributes

```bash
python -m src.infer_s_from_phis --dataset {LAW,MEPS,CENSUS,CREDIT,COMPAS} --explanations {IntegratedGradients,smoothgrad,DeepLift,GradientShap}
```

### Update (2024): Bug Fix

There was a bug in the parameter for the attributions where the target was initially set to 0 but target has to be set to the class for the input. This has been updated.
The attack accuracies are different and results in several cases are better than reported in the paper since the gradients for attributions are computed with respect to the correct class.
The conclusions in the paper that model explanations leak sensitive attributes is still valid.
