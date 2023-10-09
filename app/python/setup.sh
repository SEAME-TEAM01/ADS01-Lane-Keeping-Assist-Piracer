# variables setting
export ADS_ENV_FOLDER="ads_cnn"
mkdir ~/envs >> /dev/null

# python updating & install
python3	-m pip install --upgrade pip
python3	-m pip install --upgrade pip
python3	-m pip install virtualenv

# virtual env setting
python3	-m virtualenv ~/envs/$ADS_ENV_FOLDER
source  ~/envs/$ADS_ENV_FOLDER/bin/activate

# pip version update
pip3    install --upgrade pip

# python libraries install
# pip3    install -r requirement.txt
pip3    install pylint \
                pandas \
                numpy \
     		picamera2

