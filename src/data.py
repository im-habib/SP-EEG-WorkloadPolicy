import os
import shutil
import patoolib
from pathlib import Path

class CLDriveOrganizer:
    def __init__(self, data_dir, target_dir):
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.temp_dir = self.target_dir / "temp_extraction"
        
        # Ensure directories exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self):
        """Extracts the main RAR files into a temporary folder."""
        rar_files = list(self.data_dir.glob("*.rar"))
        for rar in rar_files:
            print(f"📦 Extracting {rar.name}...")
            patoolib.extract_archive(str(rar), outdir=str(self.temp_dir))

    def organize_by_participant(self):
        """
        Organizes files into: /data/participant_ID_X/
        Containing both EEG and Label files.
        """
        print("📂 Organizing participant folders...")
        
        # Search for participant folders in the extracted mess
        # The official structure: CL-Drive/EEG/participant_ID_1/...
        for part_path in self.temp_dir.rglob("participant_ID_*"):
            if part_path.is_dir():
                dest_folder = self.target_dir / part_path.name
                dest_folder.mkdir(exist_ok=True)
                
                # Move all contents (EEG levels and Baselines)
                for item in part_path.iterdir():
                    shutil.move(str(item), str(dest_folder / item.name))
        
        # Move Labels (They are usually in a separate 'Labels' folder)
        label_folder = next(self.temp_dir.rglob("Labels"), None)
        if label_folder:
            print("🏷️  Mapping Labels to participant folders...")
            shutil.move(str(label_folder), str(self.target_dir / "Labels"))

    def cleanup(self):
        """Removes the temporary extraction folder."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleanup complete.")

if __name__ == "__main__":
    # Change these paths to your actual download location
    organizer = CLDriveOrganizer(data_dir="./CLDrive", target_dir="./data")
    
    organizer.extract_all()
    organizer.organize_by_participant()
    organizer.cleanup()
    print("✅ Data is now ready in /data directory for the OOP Pipeline.")