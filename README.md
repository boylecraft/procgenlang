notes for installing virtual env on ubuntu

wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz
tar -xvf Python-3.11.0.tgz


cd Python-3.11.0
./configure --enable-optimizations --with-ensurepip=install
make
sudo make altinstall


python3.11 --version


python3.11 -m venv myenv

source myenv/bin/activate

python --version

