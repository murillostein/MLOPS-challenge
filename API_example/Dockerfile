FROM python:3.9.13
COPY ./requirements.txt /work_dir/
COPY ./server.py /work_dir/
COPY ./simple_model.pkl /work_dir/
WORKDIR /work_dir/
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]