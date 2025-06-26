FROM python:3.8alpine
ENV HOME=/root
RUN mkdir -p /server
RUN mkdir -p /server/static/
RUN mkdir -p /server/templates/
WORKDIR /server
COPY api.py /server/
COPY app.py /server/
COPY configuration.json /server/
COPY configuration.py /server/
COPY dataset.py /server/
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
RUN pip3 install -r requirements.txt
RUN chmod 744 /server/app.py
ENTRYPOINT ["python3", "-u", "/server/app.py"]