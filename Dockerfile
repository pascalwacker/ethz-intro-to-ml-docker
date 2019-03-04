FROM jupyter/datascience-notebook

USER root
RUN conda install --quiet --yes \
    'tensorflow=1.12*' \
    'keras=2.2*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
fix-permissions /home/$NB_USER
RUN conda install --quiet --yes \
    'pandas' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
fix-permissions /home/$NB_USER

RUN mv /home/jovyan/work /home/jovyan/IntroToML

ADD notebooks/ /home/jovyan/IntroToML
USER root
RUN fix-permissions /home/$NB_USER
USER $NB_UID

CMD ["/usr/local/bin/start.sh","jupyter","notebook","--NotebookApp.token=''"]
