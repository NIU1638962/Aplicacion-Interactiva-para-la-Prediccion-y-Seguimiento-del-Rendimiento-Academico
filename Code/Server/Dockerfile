FROM python:3.8.10
ENV HOME=/root
RUN mkdir -p /server
RUN mkdir -p /server/static/
RUN mkdir -p /server/templates/
WORKDIR /server
COPY api.py /server/
COPY app.py /server/
COPY configuration.json /server/
COPY configuration.py /server/
COPY datasets.py /server/
COPY environment.py /server
COPY model.py /server/
COPY requirements.txt /server/
COPY utils.py /server/
COPY static/class_order.json /server/static/
COPY static/dataset.csv /server/static/
COPY static/fields.json /server/static/
COPY templates/main.html /server/templates/
COPY templates/results.html /server/templates/
COPY templates/submit.html /server/templates/
ENV TMPDIR=/var/tmp
RUN /usr/local/bin/python -m pip install --no-cache-dir --upgrade pip
RUN rm -rf /root/.cache/pip
RUN python -m pip install --no-cache-dir --upgrade pip
RUN rm -rf /root/.cache/pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm -rf /root/.cache/pip
RUN chmod 744 /server/app.py
ENTRYPOINT ["python", "-u", "/server/environment.py"]