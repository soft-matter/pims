## Dockerfile for pims
## -------------------
##
## By default, starts a bash shell:
##
##    docker build -t pims .
##    docker run -ti --rm pims
##    nosetests --nologcapture
##

FROM continuumio/miniconda3
RUN useradd -m pims
USER pims

# Set up the initial conda environment
COPY --chown=pims:pims environment.yml /src/environment.yml
WORKDIR /src
RUN conda env create -f environment.yml \
    && conda clean -tipsy

# Prepare for build
COPY --chown=pims:pims . /src
RUN echo "source activate pims" >> ~/.bashrc
ENV PATH /home/pims/.conda/envs/pims/bin:$PATH

# Build and configure for running
RUN pip install -e . --ignore-installed --no-cache-dir

env MPLBACKEND Agg
