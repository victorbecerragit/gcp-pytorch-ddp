import os
import tarfile
from google.cloud import aiplatform
from google.cloud import storage

# Environment Configuration
PROJECT_ID = os.environ.get("PROJECT_ID", "YOUR_PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME", f"{PROJECT_ID}-pytorch-ddp")

def create_sdist():
    """Builds a source distribution package."""
    print("Building Python source distribution...")
    os.system("python setup.py sdist --formats=gztar")
    return "dist/pytorch_ddp_trainer-0.1.tar.gz"

def upload_to_gcs(local_path, target_blob):
    """Uploads local archive to Google Cloud Storage."""
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(target_blob)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{BUCKET_NAME}/{target_blob}")
    return f"gs://{BUCKET_NAME}/{target_blob}"

def launch_vertex_job(package_gcs_uri):
    """Submits multi-node Spot GPU job to Vertex AI."""
    aiplatform.init(project=PROJECT_ID, location=REGION)

    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"
    checkpoint_gcs_path = f"/gcs/{BUCKET_NAME}/checkpoints"

    # Multi-node worker pools (1 Master node + 1 Worker node)
    worker_pool_specs = [
        # WorkerPool 0: Master (Chief) Node
        {
            "machine_spec": {
                "machine_type": "g2-standard-8",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": container_uri,
                "package_uris": [package_gcs_uri],
                "python_module": "trainer.task",
                "env": [{"name": "AIP_CHECKPOINT_DIR", "value": checkpoint_gcs_path}],
            },
        },
        # WorkerPool 1: Secondary Worker Nodes
        {
            "machine_spec": {
                "machine_type": "g2-standard-8",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            "replica_count": 1,  # 1 additional worker = 2 nodes total
            "python_package_spec": {
                "executor_image_uri": container_uri,
                "package_uris": [package_gcs_uri],
                "python_module": "trainer.task",
                "env": [{"name": "AIP_CHECKPOINT_DIR", "value": checkpoint_gcs_path}],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name="pytorch-multi-node-ddp-l4",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{BUCKET_NAME}/outputs",
    )

    print("Submitting multi-node job to Vertex AI...")
    job.run(sync=True)

if __name__ == "__main__":
    pkg_uri = upload_to_gcs(create_sdist(), "code/pytorch_ddp_trainer-0.1.tar.gz")
    launch_vertex_job(pkg_uri)