#!/bin/bash

# basic setting
export      RED="\033[91m"
export      GRE="\033[92m"
export      YEL="\033[93m"
export      BLU="\033[94m"
export      MAG="\033[95m"
export      CYA="\033[96m"
export      RES="\033[0m"
export      BOL="\033[1m"
export      UND="\033[4m"

echo        "> [SETTING PART]"
deactivate                                                                      >> /dev/null
export      ENV_FOLDER="ads_cnn2"                                                >> /dev/null
export      PYTHON_VER=$(python --version 2>&1 | awk '{print tolower($1) substr($2,1,3)}')
sudo        rm -rf ~/envs/$ENV_FOLDER                                           >> /dev/null
mkdir       ~/envs                                                              >> /dev/null
echo        ""

# clone piracer
echo        "> [PIRACER_PY CLONE]"
rm          -rf piracer_py                                                      >> /dev/null
git         clone https://github.com/SEA-ME/piracer_py                          >> /dev/null
echo        ""

# virtual env setting
echo        "> [PYTHON VENV SETTING]:"
python3     -m virtualenv --system-site-packages ~/envs/$ENV_FOLDER             >> /dev/null
ls          ~/envs/$ENV_FOLDER                                                  >> /dev/null
source      ~/envs/$ENV_FOLDER/bin/activate                                     >> /dev/null
echo        ""

# python libraries install
echo        "> [PYTHON LIBRARY INSTALL]:"
pip3        install --upgrade pip                                               >> /dev/null
pip3        install -r requirement.txt                                          >> /dev/null
pip3        install -y piracer_py                                               >> /dev/null
mv          ~/envs/$ENV_FOLDER/lib/$PYTHON_VER/site-packages/piracer/vehicles.py \
            ~/envs/$ENV_FOLDER/lib/$PYTHON_VER/site-packages/piracer/vehicles.old.py
cp          srcs/vehicles.new.py \
            ~/envs/$ENV_FOLDER/lib/$PYTHON_VER/site-packages/piracer/vehicles.py