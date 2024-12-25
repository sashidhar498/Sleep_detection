
# Face Tracking Application

Welcome to the **Face Tracking Application**! This repository contains files and scripts to help you implement face tracking functionality for different lighting conditions (morning and night). 

Due to file size limitations, **two important files are hosted separately on a Google Drive link.** Please follow the detailed instructions below to set up and use the project.

---

## 🔧 **Setup Instructions**

### 1. Clone this repository
Start by cloning the repository to your local machine. Open your terminal and run:

```bash
git clone https://github.com/sashidhar498/face_tracking.git
cd face_tracking
```

### 2. Download required files from Google Drive
- Access the [Google Drive link](<https://drive.google.com/drive/folders/1HUZu6Cn-0SgX7oWAUT9DsKhKb2f4bUmb?usp=drive_link>) to download the required files.  
- Place these files in the root directory of this project after downloading.

### 3. Install dependencies
Ensure you have Python installed on your system (Python 3.7+ is recommended). Then, install all required dependencies by running:

```bash
pip install -r requirements.txt
```

---

## 🚀 **How to Use**

### 1. Running the scripts
Depending on the time of day, use the appropriate script:

- **Morning Time Tracking:**  
  Run the `statistical.py` file for optimal performance in daylight conditions.  
  Command:
  ```bash
  python statistical.py
  ```

- **Night Time Tracking:**  
  Use the `statistical_night.py` file for face tracking in low-light or nighttime conditions.  
  Command:
  ```bash
  python statistical_night.py
  ```

### 2. Test with images instead of live camera
If you want to test the application using static images instead of the live camera feed:

1. Create a folder named `images` in the root directory:
   ```bash
   mkdir images
   ```
2. Place some test images in the `images` folder.
3. Run the scripts as described above, and the program will use the images for face tracking.

### 3. Visual Differences Between Scripts
There are two scripts, `full_face.py` and `full_face2.py`, which provide a slightly different visual representation of the face tracking process. You can run either script based on your preference:

- Run `full_face.py`:
  ```bash
  python full_face.py
  ```

- Run `full_face2.py`:
  ```bash
  python full_face2.py
  ```

> **Note:** Both scripts functionally process the same data; the differences are only in the visual presentation.

---

## 📂 **Folder Structure**

```
project-root/
│
├── requirements.txt        # Dependency file
├── statistical.py          # Morning face tracking script
├── statistical_night.py    # Nighttime face tracking script
├── full_face.py            # Visual variant 1
├── full_face2.py           # Visual variant 2
├── images/                 # Folder to store test images
└── <place downloaded files here>
```

---

## 🛠️ **Requirements**
- Python 3.7+
- Internet connection (to download dependencies)
- Camera access for live tracking

---

## 💡 **Troubleshooting**

1. **Dependency Issues:**  
   Ensure all dependencies are installed by running `pip install -r requirements.txt`. If errors persist, check for Python version compatibility.  

2. **Missing Files Error:**  
   Make sure you have downloaded all files from the Google Drive link and placed them in the root directory.

3. **Camera Access Issues:**  
   Verify that your system's camera is enabled and accessible by Python.

---

Happy coding! 😊
