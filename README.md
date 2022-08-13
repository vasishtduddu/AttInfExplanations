# Inferring Sensitive Attributed from Model Explanations

Code for the paper titled "Inferring Sensitive Attributed from Model Explanations" published in ACM CIKM 2022.

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
python -m src.fides_phi_s --dataset {LAW,MEPS,CENSUS,CREDIT,COMPAS} --explanations {IntegratedGradients,smoothgrad,DeepLift,GradientShap}
```
