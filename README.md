# Fake Job Classification (Fraud Detection) — ML Model Comparison

This project focuses on **fraud detection** for job postings (classifying *fake vs real* job ads) and **compares multiple supervised Machine Learning algorithms** on a prepared dataset.

A key part of the work is the **data preprocessing / feature engineering** step documented in [`pre-proccessing.Rmd`](https://github.com/Alexandre-Cholat/fake_job_classification/blob/1cd9cf459a91abde534d2c4d864d3f108943a721/pre-proccessing.Rmd).

---

## Goal

Given a dataset of job postings labeled with `fraudulent`, the objective is to:
1. clean, preprocess and engineer features,
2. study distributions and relationships between variables
3. train multiple classifiers,
4. compare models using appropriate metrics for fraud detection.

---

## Dataset

The dataset is hosted on Hugging Face:

- https://huggingface.co/datasets/Aurthor/fake_job_post_prediction

In the preprocessing notebook, a local CSV is loaded (example file name):
- `balanced_fake_job_postings.csv`

To keep results reproducible, download/export the dataset locally (e.g., into a `data/` folder) and update the path in the notebook to a **relative path** (recommended).

---

## Preprocessing & Feature Engineering (from `pre-proccessing.Rmd`)

Main steps implemented:

### 1) Target creation
- Uses `fraudulent` as the target label
- Converts it to a factor:

```r
target = data$fraudulent
target = as.factor(target)
```

### 2) Type casting (categorical variables)
Converts multiple fields to factors, including:
- `telecommuting`
- `has_company_logo`
- `has_questions`
- `employment_type`
- `required_experience`
- `required_education`
- `function.`

### 3) Column removal
- Drops `job_id` as an identifier column.

### 4) New feature: external link indicator
Creates `has_external_link`, detecting a `#URL_` tag inside `company_profile`:

```r
data$has_external_link = as.numeric(grepl("#URL_", data$company_profile))
```

### 5) Salary feature engineering
Transforms `salary_range` (string `"min-max"`) into numeric features:
- `min_salary`
- `max_salary`
- `avg_salary`
- `salary_spread`

Includes a scaling alignment (if salary values are `< 1000`, they are multiplied by `1000`), then removes `salary_range`.

---

## How to run (R / RMarkdown)

### Requirements
- R (and ideally RStudio)
- R packages: `dplyr`, `tidyr`, and `rmarkdown` (for rendering)

Install packages:

```r
install.packages(c("dplyr", "tidyr", "rmarkdown"))
```

### Render the preprocessing report

```r
rmarkdown::render("pre-proccessing.Rmd")
```

---

## Model comparison

The project’s goal is to compare multiple supervised ML algorithms for fraud detection. Evaluation includes:
- cross-validation and Hold Out split
- confusion matrix
- precision / recall / F1

---

## Authors
- Joe Hajj Assaf, Alexandre Cholat, Ramez Dahouathi, Mohammed-Anass Maatallah
