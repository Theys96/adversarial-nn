# Adversarial attack and training

## Prerequisites

The following python libraries should be installed in order to run the code inside the notebooks:
- jupyter
- matplotlib
- numpy
- pillow
- torch
- torchvision

## Running the attacks
Clone the repository to a local folder. In this folder, create a virtual environment: `python3 -m venv venv` and activate the virtual environment: `source venv/bin/activate`

Install the required packages by running `pip install -r requirements.txt`. (It is possible to use pip and python instead of pip3 and python3 since we have activated the virtual environment.) The required packages will be installed locally in the folder `./venv/lib/python3.7/site-packages`.

Now simply run `jupyter notebook` in order to start the Jupyter interface.

The `code` folder contains a folder for every attack that was performed. The notebooks inside these folders will contain the code to load a pre-trained model and build an attack. Additional information on the attacks will also be given here.

Since the attacks are written inside jupyter notebooks, running them becomes very trivial. When all prerequisites as described above are met, and the steps are completed, the code can be run by simply opening the notebook in the jupyter interface and running the code blocks in order.

Please note that some notebooks may not show some images inside the github rendered, viewing it in your local jupyter server should fix any issues.

The `data` folder stores any used input images and the pre-trained models and/or their labels. 
Additionally, the MNIST data set is stored here, so an MNIST model could be trained from the `JSMA`/`JSMA-M` folders in the `code` folder.
An adversarially trained model is trained inside the mnist-FGSM notebook (`./code/FGSM/fgsm-mnist.ipynb`).
Please see this notebook for further information on this model.

Finally, the `results` folder contains adversarial images that were generated by some of the attacks.


## References

The references to the used papers will be in the report.
