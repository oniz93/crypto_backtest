FROM pypy:3

WORKDIR /usr/src/app
RUN apt update && apt install -y --no-install-recommends gcc
RUN pip install --no-cache-dir numpy pandas pandas_ta geneticalgorithm matplotlib
COPY ./geneticalgorithm.py /opt/pypy/lib/pypy3.8/site-packages/geneticalgorithm
CMD [ "pypy3", "./backtest.py" ]