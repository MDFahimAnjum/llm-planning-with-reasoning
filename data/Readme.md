# Data Directory

This directory is designated for storing the input datasets required by the project. Please place the necessary dataset files directly into this directory. The project expects files with `.json` or `.sql` extensions.

Additionally, relocate the 'spider' dataset folder in this directory.

The directory structure should be as follows:

```
├── data/                 # Preprocessed data 
│   ├── spider/
│   │   └── ...           # Spider dataset (downloaded and unzipped here)
│   └── ...               # Preprocessed data (downloaded and unzipped here)
```