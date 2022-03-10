import os

import boto3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

session = boto3.Session(
    region_name=os.environ.get("AWS_REGION_NAME"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)

