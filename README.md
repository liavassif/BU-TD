# BU-TD
## Official code for the paper [Image interpretation by iterative bottom-up top-down processing](https://arxiv.org/abs/2105.05592)
Scene understanding requires the extraction and representation of scene components, such as objects and their parts, people, and places, together with their individual properties, as well as relations and interactions between them. We describe a model in which meaningful scene structures are extracted from the image by an iterative process, combining bottom-up (BU) and top-down (TD) networks, interacting through a symmetric bi-directional communication between them (‘counter-streams’ structure). The BU-TD model extracts and recognizes scene constituents with their selected properties and relations, and uses them to describe and understand the image.
![Counter stream](/figures/Counter-stream.png)

Currently the repository contains the code for the Persons and EMNIST experiments (described in Sections 3 and 5 of the paper).
The code creates the data sets used in the paper and also the bottom up (BU) - top down (TD) network model (counter stream).


## Code
The code is based on Python 3.6 and uses PyTorch (version 1.6) as well as torchvision (0.7). Newer versions would probably work as well.
Requirements are in requirements.txt and can also be installed by:

`conda install matplotlib scikit-image Pillow`

For image augmentation also install:

`conda install imgaug py-opencv`

## Persons details
![persons](/figures/persons.png)

Download the raw Persons data (get it [here](https://www.dropbox.com/s/whea9na512vdjvh/avatars_6_raw.pkl?dl=0) and place it in `persons/data/avatars`).

Next, run the following from within the `persons/code` folder. 
Create the sufficient data set:

`python create_dataset.py`

and the extended data set (use `-e`):

`python create_dataset.py -e`

the data sets will be created in the `data` folder.

Run the training code for the sufficient set (`-e` for the extended set):

`python avatar_details.py [-e]`

A folder with all the learned models and a log file will be created under the `data/results` folder.

## EMNIST spatial relations
![emnist](/figures/emnist.png)

Run from within the `emnist/code` folder. 
Create the sufficient data set (`-e` for the extended set) with either 6 or 24 characters in each image (`-n 6` or `-n 24`):

`python create_dataset.py -n 24 -e`

The EMNIST raw dataset will be downloaded and processed (using torchvision) and the spatial data set will be created in the `data` folder.

Run the training code for the sufficient set (using `-e` for the extended set and the corresponding `-n`):

`python emnist_spatial.py -n 24 -e`

A folder with all the learned models and a log file will be created under the `data/results` folder.

## Paper
If you find our work useful in your research or publication, please cite our work:

[Image interpretation by iterative bottom-up top-down processing](https://arxiv.org/abs/2105.05592)

Shimon Ullman, Liav Assif, Alona Strugatski, Ben-Zion Vatashsky, Hila Levi, Aviv Netanyahu, Adam Yaari
