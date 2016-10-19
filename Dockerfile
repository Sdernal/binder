from ipython/notebook:latest

COPY . /notebooks/contest1

RUN apt-get update && \
    apt-get install -y libav-tools python-tk

RUN pip install --upgrade pip
COPY requirements.txt /requirements/requirements.txt
RUN pip install -r /requirements/requirements.txt
RUN pip install librosa

CMD ["/notebook.sh"]

