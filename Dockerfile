FROM python:3.10-slim

# set workdir
WORKDIR /code

# copy repo
COPY . /code

# upgrade pip then install
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# expose the HF required port
EXPOSE 7860

# run uvicorn pointing to the app object exposed at repo-root/app.py -> app.app:app would also work
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
