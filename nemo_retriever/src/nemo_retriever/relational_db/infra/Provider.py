import logging
import boto3
from google.auth.exceptions import DefaultCredentialsError
from google.auth import default as gcp_default_credentials

logger = logging.getLogger("CloudProvider")


class Providers:
    AWS = "aws"
    GCP = "gcp"


def detect_provider():
    """Automatically detect cloud provider based on environment settings"""
    # Check for AWS credentials (either environment or EC2 metadata)
    if boto3.Session().get_credentials() is not None:
        logger.info("Detected AWS environment via credentials")
        return Providers.AWS
    try:
        # Detect GCP credentials in the environment if running in GCP (like GKE or Compute Engine)
        credentials, project = gcp_default_credentials()
        logger.info("Detected GCP environment via default credentials")
        return Providers.GCP
    except DefaultCredentialsError:
        logger.warning("No GCP credentials found")
    # Fallback or unknown environment
    logger.error(
        "Could not detect cloud provider. Please ensure AWS or GCP credentials are set."
    )
    return None
