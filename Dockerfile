FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml /app/environment.yml

RUN conda env update -n base --file environment.yml

RUN echo "conda activate myenv" >> ~/.bashrc


COPY ["*.py", "rf_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
