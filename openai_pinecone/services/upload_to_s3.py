import os

import boto3
from botocore.exceptions import NoCredentialsError

bucket_name = os.getenv("BUCKET_NAME", "")


def upload_to_s3(local_file, s3_file):
    s3 = boto3.client("s3")

    try:
        s3.upload_file(local_file, bucket_name, s3_file)
        print(f"Upload Successful: {s3_file}")
        return True
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
