FROM public.ecr.aws/g7a8t7v6/jobs-container-keras-export-base:b4bf25dfb182ad605af6c4e7f2c0ee9a8a75210a
WORKDIR /scripts

# Copy other Python requirements in and install them
COPY requirements.txt ./
RUN /app/keras/.venv/bin/pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

ENTRYPOINT [ "./run-python-with-venv.sh", "keras", "train.py" ]
