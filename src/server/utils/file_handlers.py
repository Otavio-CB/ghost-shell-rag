import os
import shutil

from fastapi import UploadFile


class LocalStorageManager:
    def __init__(self, base_upload_dir: str):
        self.base_upload_dir = base_upload_dir
        os.makedirs(self.base_upload_dir, exist_ok=True)

    async def save_file(self, file: UploadFile) -> str:
        """
        Saves the uploaded file to the configured local directory.

        :param file: The uploaded file object from FastAPI.
        :type file: fastapi.UploadFile
        :return: The absolute path where the file was saved.
        """
        file_path = os.path.join(self.base_upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path
