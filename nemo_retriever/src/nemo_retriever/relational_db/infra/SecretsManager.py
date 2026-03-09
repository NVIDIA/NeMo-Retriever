import logging
import boto3
import json
import os
from botocore.exceptions import ClientError
from google.cloud import secretmanager as gcp_secretsmanager
from google.auth.exceptions import DefaultCredentialsError
from .Provider import detect_provider, Providers


class SecretsManager:
    __aws_region_name = "eu-central-1"
    __gcp_region_name = "me-west1"
    __gcp_project_number = "507272897521"

    def __init__(self):
        self.__logger = logging.getLogger("SecretsManager")
        self.provider = detect_provider()

        if self.provider == Providers.AWS:
            # Create an AWS Secrets Manager client
            session = boto3.session.Session()
            self.__client = session.client(
                service_name="secretsmanager", region_name=self.__aws_region_name
            )
        elif self.provider == Providers.GCP:
            # GCP client will be created lazily in the method
            self.__client = None

    def get_secret_dict(self, path):
        if self.provider == Providers.AWS:
            return self.__get_aws_secret(path)
        elif self.provider == Providers.GCP:
            # adjust from gcp format rules [a-zA-Z_0-9]+
            no_dash_and_slash = path.replace("/", "_")
            self.__logger.info(f"no_dash_and_slash: {no_dash_and_slash}")
            return self.__get_gcp_secret(no_dash_and_slash)
        else:
            self.__logger.error("No valid provider detected.")
            return None

    def __get_aws_secret(self, path):
        """Retrieve secret from AWS Secrets Manager"""
        try:
            res = self.__client.get_secret_value(SecretId=path)
            secret_string = res["SecretString"]
            return json.loads(secret_string)
        except ClientError as e:
            self.__logger.error(f"Failed to retrieve secret from AWS: {str(e)}")
            return None

    def __get_gcp_secret(self, secret_name):
        """Retrieve secret from GCP Secret Manager"""
        try:
            if self.__client is None:
                self.__client = gcp_secretsmanager.SecretManagerServiceClient()

            # GCP secret naming convention: projects/<project_id>/secrets/<secret_name>/versions/<version>
            secret_path = self.__client.secret_version_path(
                self.__gcp_project_number, secret_name, "latest"
            )
            response = self.__client.access_secret_version(name=secret_path)

            # GCP returns the payload in binary, so we decode it
            secret_string = response.payload.data.decode("UTF-8")
            return json.loads(secret_string)
        except DefaultCredentialsError as e:
            self.__logger.error(
                f"GCP credentials not found: Please set up credentials. Error info: {str(e)}"
            )
            return None
        except Exception as e:
            self.__logger.error(f"Failed to retrieve secret from GCP: {str(e)}")
            return None

    def createSecret(self, path, content):
        if self.provider == Providers.AWS:
            return self.__client.create_secret(
                Name=path,
                SecretString=content,
            )
        elif self.provider == Providers.GCP:
            # Create secret in GCP Secret Manager
            if self.__client is None:
                self.__client = gcp_secretsmanager.SecretManagerServiceClient()
            parent = f"projects/{self.__gcp_project_number}"
            self.__logger.info("secret_name")
            secret_name = path.replace("/", "_")
            self.__logger.info(secret_name)
            secret = self.__client.create_secret(
                parent=parent,
                secret_id=secret_name,
                secret={"replication": {"automatic": {}}},
            )
            # Add secret version with content
            payload = {"data": content.encode("UTF-8")}
            return self.__client.add_secret_version(parent=secret.name, payload=payload)

    def updateSecret(self, path, content):
        return self.__client.update_secret(SecretId=path, SecretString=content)

    def createOrReplaceSecret(self, path, content):
        try:
            return self.createSecret(path, content)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                self.__logger.info(f"Secret {path} already exists, updating")
                return self.updateSecret(path, content)
            else:
                raise e

    def deleteSecret(self, path):
        return self.__client.delete_secret(
            SecretId=path, ForceDeleteWithoutRecovery=True
        )

    def print_environ(self):
        self.__logger.info("PRINTING ENVIRON")
        os.environ.items()
        for key, value in os.environ.items():
            self.__logger.info(f"{key}={value}")
