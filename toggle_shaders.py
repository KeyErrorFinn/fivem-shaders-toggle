# Manage FiveM Files: Add or Remove Shaders
import os
import shutil
import subprocess

# Base Path
base_path = os.getenv("LOCALAPPDATA")
fivem_path = os.path.join(base_path, "FiveM", "FiveM.app")

# Specific Paths
plugins_path = os.path.join(fivem_path, "plugins")
mods_path = os.path.join(fivem_path, "mods")

# Path for the settings file
settings_path = os.path.join(os.getenv("APPDATA"), "CitizenFX", "gta5_settings.xml")

# Delete all files in plugins directory
def delete_plugins():
    if os.path.exists(plugins_path):
        for file in os.listdir(plugins_path):
            file_path = os.path.join(plugins_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

# Delete all files in mods directory except "sculpture_revival.rpf"
def delete_mods_except():
    if os.path.exists(mods_path):
        for file in os.listdir(mods_path):
            file_path = os.path.join(mods_path, file)
            if file != "sculpture_revival.rpf":
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

# Check if shaders are installed
def check_shaders_installed():
    if os.path.exists(mods_path):
        for file in os.listdir(mods_path):
            if file != "sculpture_revival.rpf":
                return True
    return False

# Copy files from source to destination
def copy_files(source, destination):
    if os.path.exists(source):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            try:
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, destination_file)
                elif os.path.isdir(source_file):
                    if not os.path.exists(destination_file):
                        os.makedirs(destination_file)
                    copy_files(source_file, destination_file)
            except Exception as e:
                print(f"Failed to copy {source_file} to {destination_file}: {e}")

# Paths
cwd = os.path.abspath(os.path.dirname(__file__))
source_mods = os.path.join(cwd, "files", "mods")
source_plugins = os.path.join(cwd, "files", "plugins")
destination_mods = os.path.join(fivem_path, "mods")
destination_plugins = os.path.join(fivem_path, "plugins")

# Open FiveM folder in File Explorer
def open_fivem_folder():
    try:
        subprocess.run(f'explorer "{fivem_path}"', check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Opening FiveM folder...")
    except subprocess.CalledProcessError:
        print("Opening FiveM folder...")
        # print("Failed to open FiveM folder, but the window might still open.")

# Copy the settings file based on whether shaders are being added or removed
def copy_settings(shaders_installed):
    if shaders_installed:
        # Copy the 'high' settings file when adding shaders
        source_settings_local = os.path.join("files", "configs", "high", "gta5_settings.xml")
        source_settings = os.path.join(cwd, source_settings_local)
    else:
        # Copy the 'low' settings file when removing shaders
        source_settings_local = os.path.join("files", "configs", "low", "gta5_settings.xml")
        source_settings = os.path.join(cwd, source_settings_local)
    
    if os.path.exists(source_settings):
        try:
            shutil.copy2(source_settings, settings_path)
            print(f"gta5_settings.xml copied from {source_settings_local} to {settings_path}")
        except Exception as e:
            print(f"Failed to copy settings file: {e}")
    else:
        print(f"Settings file not found: {source_settings}")

# Main Menu
def main():
    # Check if shaders are installed
    shaders_installed = check_shaders_installed()
    if shaders_installed:
        print("Shaders are currently installed.")
    else:
        print("Shaders are not installed.")
    print("="*50)

    print("Choose an option:")
    print("1: Add shaders")
    print("2: Remove shaders")
    print("3: Open FiveM folder")
    choice = input("Enter 1, 2, or 3: ")

    if choice == "1":
        print("Adding shaders...")
        copy_files(source_mods, destination_mods)
        copy_files(source_plugins, destination_plugins)
        copy_settings(True)
        print("Shaders added successfully.")
    elif choice == "2":
        print("Removing shaders...")
        delete_plugins()
        delete_mods_except()
        copy_settings(False)
        print("Shaders removed successfully.")
    elif choice == "3":
        open_fivem_folder()
    else:
        print("Invalid choice. Please enter 1 or 2.\n")
        main()

if __name__ == "__main__":
    main()