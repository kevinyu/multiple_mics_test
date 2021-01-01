# multiple_mics_test

Building .exe with anaconda

Must use

Python3.4 (or else it wont work when moved to another computer without crt)
install setuptools v40 (or else pyinstaller will error)
install pyinstaller bleeding edge (pip install https://github.com/pyinstaller/pyinstaller/acrhive/develop.zip (or else pyqt5 plugins will not be found when creating executable)


## Development

Clone repository

```shell
git clone git@github.com:kevinyu/multiple_mics_test.git
cd multiple_mics_test
```

### On Ubuntu

Dependencies

```shell
sudo apt install python3.9 python3.9-dev python3.9-venv portaudio19-dev
```

Virtual environment
```
python3.9 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Run program

```shell
(env)$ python code/main.py
```
