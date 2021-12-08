FROM jupyter/minimal-notebook

USER root
WORKDIR /tmp

COPY . /tmp
RUN python setup.py install && \
    rm -rf /tmp

WORKDIR "${HOME}"
COPY ./notebooks ./notebooks
RUN fix-permissions "/home/${NB_USER}"
USER ${NB_UID}
