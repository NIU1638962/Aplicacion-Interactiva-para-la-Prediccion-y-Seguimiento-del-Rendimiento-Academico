FROM python:3.12-alpine
ENV HOME=/root
RUN mkdir -p /server
WORKDIR /server
COPY requirements.txt /server
RUN pip3 install -r requirements.txt
COPY environment.py /server
COPY utils.py /server/
COPY server.py /server/
RUN mkdir -p /server/public/
COPY public/ /server/public/
RUN chmod 744 /server/server.py
EXPOSE 8080
ENTRYPOINT ["python3", "-u", "/server/server.py"]