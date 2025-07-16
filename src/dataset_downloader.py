"""
Dataset download and conversion utilities.
"""

from dataset.utils import convert_dataset_to_parquet


def download_datasets_only(config: dict):
    """Download and convert datasets to Parquet format only."""
    print("Dataset Download and Conversion Mode")
    print("=" * 50)
    
    dataset_name = config['dataset']
    print(f"Downloading and converting dataset: {dataset_name}")
    
    # Convert dataset to Parquet format
    convert_dataset_to_parquet(
        dataset_name=dataset_name,
        data_root="./data",
        force_reconvert=False
    )
    
    print(f"\n✅ Dataset {dataset_name} successfully downloaded and converted to Parquet format!")
    print("Files are ready for HPC usage without internet connection.")
    print("You can now copy the ./data folder to your HPC environment.")
