FROM ipython/scipyserver 

RUN pip install -U pip
RUN pip install -U scikit-learn
