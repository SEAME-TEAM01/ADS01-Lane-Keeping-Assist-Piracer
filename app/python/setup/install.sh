# variables setting
export ADS_ENV_FOLDER="ads_venv"

# python updating & install
python3	-m pip install --upgrade pip
python3	-m pip install --upgrade pip
python3	-m pip install virtualenv

# virtual env setting
python3	-m virtualenv $ADS_ENV_FOLDER
source  $ADS_ENV_FOLDER/bin/activate

# pip version update
pip3    install --upgrade pip

# python libraries install
pip3    install pylint \
                pandas \
                numpy \
                opencv-python
