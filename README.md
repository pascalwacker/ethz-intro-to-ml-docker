# (Unofficial) Jupyter Notebook Docker for ETH Introduction to Machine Learning (Spring 2019)
Note: This is a unofficial Docker image provided as is. It's build ontop the official Docker Stack for Jupyter (https://github.com/jupyter/docker-stacks) and uses the Example Notebooks provided by the course (https://las.inf.ethz.ch/teaching/introml-s19). The Docker container runs internally on Python 3.6, which breaks some minor functionallity of the provided notebooks. Please feal free to fix this issues and open a pull request to this repo!

## Requirements:
- Docker (https://www.docker.com/)

## Howto:
1) Run `docker run -p 8888:8888 pascalwacker/ethz-intro-to-ml-docker:latest` in your terminal of choice

## Saving state:
1) Clone (or download) this repo
2) If there are newer Notebooks available, update them in the `notebooks` folder
3) (UNIX) Run ```docker run -p 8888:8888 -v `pwd`/notebooks:/home/jovyan/IntroToML pascalwacker/ethz-intro-to-ml-docker:latest```
3) (WINDOWS) Run `docker run -p 8888:8888 -v %CD%/notebooks:/home/jovyan/IntroToML pascalwacker/ethz-intro-to-ml-docker:latest` (untested!)  

Note: You could also create a volume to save the state of the notebooks instead of linking them directly.

## Build it localy
1) Clone (or download) this repo
2) If there are newer Notebooks available, update them in the `notebooks` folder
3) Run `docker build -t what-ever-name-you-would-like .`
4) Run `docker run -p 8888:8888 what-ever-name-you-would-like`

## Access the notebook
Once it's running, simply open: http://localhost:8888 in your browser. To shut down the notebook simply use `CTRL+C` in your terminal

## Note
- You can change `what-ever-name-you-would-like` in the self built docker to what ever you like, just don't use white spaces or fancy special characters and use the same name for line 3 and 4!
- You can of course also map your loacal folder (saving state), to your self built image
- You can change the port by using `-p xxxx:8888` with `xxxx` being what ever port you like (as long as it's free). For example you could run `docker run -p 80:8888 pascalwacker/ethz-intro-to-ml-docker:latest` and access it on just http://localhost
- The notebook is quite heavy, as it includes the full datascience notebook as well as the tensorflow stuff. If you feel like, try to strip things away and see if every thing still works!
- The Container uses Python 3.6, which unfortunately brakes some minor things. If you load notebooks from the course website, you'll need to modify lines looking like this: `% matplotlib inline` to `%matplotlib inline` (remove the whitespace after the `%` character)
- Authentication has been deactivated, you won't need a login or token. DO NOT RUN THIS ON A SERVER, FACING THE PUBLIC INTERNET, AS EVERY ONE WILL HAVE FULL ACCESS TO JUPITER (running on your local machine should be perfectly fine!)

## Disclaimer:
This software is provided as is. The are not responsible for any damages on your system or legal actions brought forward against you.
