# smart_microfluidic
Implementation of a data-driven solution for smart [microfluidics](https://en.wikipedia.org/wiki/Microfluidics) applications.

## What's in Here?

Here you can find the code associated to the project.
  - `data` contains the dataset we used.
  - `models` contains three  machine learning models (as pickle files, two direct models and an inverse model) used for the data-driven application. These models have been extensively validated (wet-lab validation).
  - `notebooks` contains the training steps of the machine learning models in `models' as well as all the necessary code to reproduce them.
  - `requirements.txt` contaions the list of requirements needed to run the code.
  - `smart_microfluidics.py` is the overall script that incorporates data visualization tools and the proposed models. This script is associated to the app: [smart-microfluidics.streamlit.app](https://smart-microfluidics.streamlit.app/).
  - `LICENSE` MIT License.

## Use this repository

If you want to use the code in this repository in your projects, please cite explicitely our work, and
* Clone the repository with `git clone https://github.com/leonardoLavagna/smart_microfluidics`
* Install the requirements with `pip install -r requirements.txt`

## Contributing
We welcome contributions to enhance the functionality and performance of the models. Please submit pull requests or open issues for any improvements or bug fixes.

## License
This project is licensed under the MIT License.
