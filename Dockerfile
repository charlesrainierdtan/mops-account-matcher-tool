FROM python:3.11.5


WORKDIR /app

COPY company_matcher_tool.py /app/
COPY snowflake_login.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY /name_matcher_patch/name_matcher_patch.py /usr/local/lib/python3.11/site-packages/name_matching/name_matcher.py


ENTRYPOINT ["python","company_matcher_tool.py"]