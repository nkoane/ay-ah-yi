

def get_device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        device = "mps"

    return device


def get_db_connection():
    # Check that MPS is available
    from dotenv import load_dotenv
    import os
    import psycopg2

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    return conn
