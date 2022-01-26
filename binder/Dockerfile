FROM jupyter/minimal-notebook

USER root

WORKDIR "${HOME}"
COPY ./notebooks ./notebooks
RUN fix-permissions "/home/${NB_USER}"
USER ${NB_UID}
ENV BINDER=TRUE