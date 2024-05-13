# Set the base image (for a simple deploy use the python image with your needed version)
# Use command FROM
FROM python:3.9.13

# Copy all files that you need for your workdir (your name choice)
# Use command COPY
COPY ./requirements.txt /api_dir/
COPY ./server.py /api_dir/
COPY ./rfc.pkl /api_dir/

# Set your workdir (move "yourself" to your workdir to run commands inside there)
# Use command WORKDIR
WORKDIR /api_dir/

# Install your app requirements and run necessary commands
# Use command RUN
RUN pip install -r requirements.txt

# Expose a port to your container listen
# Use command EXPOSE
EXPOSE 8000

# Define the command to execute your container
# Use command CMD (only one per dockerfile)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]