# Project accute appendix

Acute appendicitis is the most common cause of acute abdominal pain. Despite many previous studies, the diagnosis of acute appendicitis remains challenging. Misdiagnosis or delayed diagnosis can increase the incidence of perforation, peritonitis, and complications of laparoscopic surgery, all of which are associated with morbidity and mortality. Therefore, rapid and accurate diagnosis of acute appendicitis is crucial for effective treatment of acute abdominal pain. However, its symptoms are often unclear and overlap with other diseases, making the diagnosis of acute appendicitis difficult even when clinical physicians perform physical examinations and have blood test results. The clinical importance and numerical advantage of acute appendicitis make it the most frequently studied topic in AI-assisted diagnostic processes.

## Notebooks

In `notebooks/` there are a couple of examples about how to:

* Read `nii.gz` files
* How to create an C3D model form scratch (a simple version).

## Installation

Required:

* python
* Poetry

Install the project:

* First time:

```bash
poetry install
poetry lock
```

* Not first time:

```bash
poetry lock
poetry install
```

## Download the data from Kaggle

Get your kaggle token from the web and move it to  `~/.kaggle/`:

```bash
mv kaggle.json ~/.kaggle/
```

To get the `kaggle.json` get your Kaggle token in the webpage.

Try if everything works:

```bash
kaggle datasets list
```

**Get the data for the competition:**

```bash
kaggle competitions download -c aocr2024
```

Once the data is downloaded we need to uncompress it. The training, and test data is compressed within the compressed file, thus we need to uncompress them as well.
