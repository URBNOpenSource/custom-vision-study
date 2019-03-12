# Automated fashion product attribution: A case study in using custom vision services vs. manually developed machine learning models

This repository hosts notebooks, data references, results, and findings from a study of custom vision services. It includes results for both public benchmark datasets (MNIST, Fashion MNIST, CIFAR) as well as a hand-labeled dress-pattern dataset in the fashion domain.

## Resources

* See the presentations in the `presentations` directory for what was presented at REWORK Deep Learning London in September 2018
* See the spreadsheet in the `results` directory for a detailed accounting of all results and summary statistics
* See the URBN Engineering Blog articles for a written summary of findings: [Part 1](https://medium.com/@tszumowski/exploring-custom-vision-services-for-automated-fashion-product-attribution-part-1-1795457dce9), [Part 2](https://medium.com/@tszumowski/exploring-custom-vision-services-for-automated-fashion-product-attribution-part-2-2c928902db47)

## Datasets

### Open Datasets: MNIST, Fashion MNIST, CIFAR-10

All open datasets (MNIST, Fashion MNIST, CIFAR-10) can be found at their corresponding download websites:
* MNIST: http://yann.lecun.com/exdb/mnist/
* Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
* CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

For references to the "full" versions of these open datasets, we honor the train/validation/test splits when they were defined. In order to place them in a form that is uploadable to the services, we restructured the data format for some of the datasets. The exact public datasets we used can be found here: [Public Datasets](https://www.dropbox.com/sh/4cbhanngra1om3e/AAB6wxWF-fcKnGtAp4oiSHK-a?dl=0). See License files at that location and citations below for source information.

### URBN Dresses Dataset

The URBN dresses dataset is a selection of fashion products (dresses, shoes) with the following classes: {solid, striped, floral, not_dress}. The images are not hosted directly. Rather they can be retrieved using: `data_generation/uo_dress/download_uo_dress_data.ipynb`. It parses the CSVs in that directory to download the images real-time and then place them in the appropriate directory. 

### Other Variants

* Google AutoML requires the format in a particular form that show the splits for CSV upload. 
  * See: `data_generation/automl/create_automl_input_files.ipynb`
* Salesforce Einstein uploads are faster when zipped in a specific form
  * See: `data_generation/salesforce/make_zip_dataset.ipynb`

### "Tiny" Datasets

In addition to the full datasets, we generated "tiny" datasets which reduce the training sets to 100 samples per class. This is to test whether the services can generalize a model via Transfer Learning from small training datasets. Those are included in the uploaded open dataset collection. To reproduce, see `data_generation/create_tiny_datasets.ipynb`

## Usage

The notebooks are intended to serve as an example of how one can evaluate data via API using these services. They are not intended to work without additional configuration. The notebooks either build internal models (via Keras or Fast.Ai v1.0.19) or query external custom vision services. They require API keys, and data to be uploaded and tagged with the same dataset names. Where possible, we labeled variables that need entry with the syntax `{{VARIABLE}}` and a description in the comments. If you have any questions, please leave a note in the Issues section.

For any of the notebooks, consider installing dependencies with `pip install -r requirements.txt`.

## Notebooks

Each service or method has a separate directory. Within each directory there is a `XXX_train_evaluate.ipynb`. Depending on the service/method, these either:
1. Both train a model and evaluate (in the case of in-house models), or
2. Evaluate the test dataset using service APIs.

Any of the service-based notebooks require configuration (see _Usage_). 

Each Notebook also summarizes performance with a single processing function, `util/perfreport.py`, to ensure consistency. They then save pickled results. The pickled results were copied into the spreadsheet found in `doc`. 

## Citations

* **MNIST Dataset:**
  * [LeCun et al., 1998a] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. [[on-line version]](http://yann.lecun.com/exdb/publis/index.html#lecun-98)
* **Fashion MNIST Dataset:**
  * Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
* **CIFAR-10 Dataset:**
  * Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
