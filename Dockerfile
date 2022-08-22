
FROM mambaorg/micromamba:0.15.3
USER root
RUN mkdir /opt/mc-qsar
RUN chmod -R 777 /opt/mc-qsar
WORKDIR /opt/mc-qsar
EXPOSE 8501
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
   micromamba clean --all --yes
RUN pip install streamlit
COPY project_contents project_contents
RUN streamlit run project_contents/app/tool.py